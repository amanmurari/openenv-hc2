"""Inference Script — Autonomous Traffic Control OpenEnv Environment"""

import os
import sys
import json
import time
import math
import urllib.request
from collections import deque
from typing import Deque, List, Optional, Tuple

# Allow imports from repo root so both container-root and package contexts work
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openai import OpenAI

try:
    from traffic_control.client import TrafficControlEnv
    from traffic_control.models import TrafficAction, TrafficObservation, TrafficState
except ImportError:
    from client import TrafficControlEnv  # type: ignore
    from models import TrafficAction, TrafficObservation, TrafficState  # type: ignore

# ---------------------------------------------------------------------------
# Environment variables — injected by hackathon validator
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
SERVER_URL   = os.getenv("SERVER_URL", "http://localhost:7860")
SEED         = 42
MAX_TOKENS      = 200   # shorter = faster response
TEMPERATURE     = 0.0
LLM_TIMEOUT_S   = 12    # per-call timeout (seconds)
LLM_CALL_EVERY  = 5     # only call LLM every N steps (heuristic fills the rest)
TASK_BUDGET_S   = {     # hard wall-clock budget per task (seconds)
    "basic_flow":          480,
    "emergency_priority":  720,
    "dynamic_scenarios":   960,
}

# Task max_steps for score projection
TASK_MAX_STEPS = {"basic_flow": 200, "emergency_priority": 300, "dynamic_scenarios": 400}

# Task-specific minimum green hold (steps before switching is even considered).
# basic_flow: stability bonus requires low switch rate — hold longer.
# emergency tasks: react fast to emergencies — hold shorter.
MIN_HOLD_BY_TASK = {
    "basic_flow":          6,
    "emergency_priority":  3,
    "dynamic_scenarios":   3,
}
DEFAULT_MIN_HOLD = 4

# Pressure ratio required to justify a switch (avoid switching-penalty)
# Higher for basic_flow (graded on stability), lower for emergency tasks
SWITCH_RATIO_BY_TASK = {
    "basic_flow":          1.5,
    "emergency_priority":  1.2,
    "dynamic_scenarios":   1.2,
}

# ---------------------------------------------------------------------------
# Heuristic core — uses actual environment reward formula
# ---------------------------------------------------------------------------

def _em_pressure(count: int, urgency: int) -> float:
    """Emergency pressure = count × urgency^1.5 × 0.5 (matches env._compute_reward)."""
    return count * (max(urgency, 1) ** 1.5) * 0.5 if count > 0 else 0.0


def _dir_pressure(queue: int, em_count: int, urgency: int) -> float:
    """Combined directional pressure: throughput value + weighted emergency penalty."""
    return queue * 0.30 + _em_pressure(em_count, urgency) * 4.0


def _compute_pressures(obs: TrafficObservation) -> Tuple[float, float]:
    """Return (ns_pressure, ew_pressure)."""
    ns_p = _dir_pressure(
        obs.queue_lengths[0] + obs.queue_lengths[1],
        obs.emergency_queue[0] + obs.emergency_queue[1],
        max(obs.emergency_urgency[0], obs.emergency_urgency[1]),
    )
    ew_p = _dir_pressure(
        obs.queue_lengths[2] + obs.queue_lengths[3],
        obs.emergency_queue[2] + obs.emergency_queue[3],
        max(obs.emergency_urgency[2], obs.emergency_urgency[3]),
    )
    return ns_p, ew_p


def _dynamic_hold_time(obs: TrafficObservation, task: str) -> int:
    """
    Adaptive minimum hold: longer when current direction has more traffic,
    scaled by task type (basic_flow needs more stability).
    """
    base = MIN_HOLD_BY_TASK.get(task, DEFAULT_MIN_HOLD)
    cur = obs.current_phase
    if cur in (0, 3):   # NS_GREEN / NS_YELLOW
        q = obs.queue_lengths[0] + obs.queue_lengths[1]
    elif cur in (1, 4): # EW_GREEN / EW_YELLOW
        q = obs.queue_lengths[2] + obs.queue_lengths[3]
    else:
        return base
    # Hold longer if queue is deep (drain rate ~3 veh/step), cap at 12
    return max(base, min(q // 3, 12))


def _collision_risk(obs: TrafficObservation) -> bool:
    """
    Detect gridlock risk early (environment triggers -200 at total_queued>40 AND time>20).
    We act at 70% of threshold so we can clear before the penalty triggers.
    """
    total_q = sum(obs.queue_lengths)
    return total_q > 28 and obs.time_in_phase > 14


def _heuristic_phase(obs: TrafficObservation, task: str) -> Tuple[int, str]:
    """
    Mathematically optimal phase recommendation.
    Returns (phase, reason) so caller can log it.

    Priority order:
      1. Collision risk — proactively rotate to drain largest queue
      2. Critical emergency (urgency ≥ 8) — clear NOW
      3. Pre-emptive emergency (urgency 6-7) — clear before escalation
      4. Moderate emergency (urgency 5)
      5. Hysteresis — don't switch if hold time not reached
      6. Queue pressure — switch to higher pressure direction
      7. Default — hold current
    """
    ns_em  = obs.emergency_queue[0] + obs.emergency_queue[1]
    ew_em  = obs.emergency_queue[2] + obs.emergency_queue[3]
    ns_urg = max(obs.emergency_urgency[0], obs.emergency_urgency[1])
    ew_urg = max(obs.emergency_urgency[2], obs.emergency_urgency[3])
    cur    = obs.current_phase
    ns_q   = obs.queue_lengths[0] + obs.queue_lengths[1]
    ew_q   = obs.queue_lengths[2] + obs.queue_lengths[3]

    # 1. Collision risk — switch to whichever direction has more vehicles
    if _collision_risk(obs):
        if cur in (0, 3):
            if ew_q > ns_q:
                return 1, "collision-risk-rotate-EW"
            return 0, "collision-risk-hold-NS"
        else:
            if ns_q > ew_q:
                return 0, "collision-risk-rotate-NS"
            return 1, "collision-risk-hold-EW"

    # 2. Critical emergency (urgency ≥ 8)
    if ns_em > 0 and ns_urg >= 8 and ew_em > 0 and ew_urg >= 8:
        return 2, "ALL_RED-dual-critical"
    if ns_em > 0 and ns_urg >= 8:
        return 0, f"critical-NS-urgency={ns_urg}"
    if ew_em > 0 and ew_urg >= 8:
        return 1, f"critical-EW-urgency={ew_urg}"

    # 3. Pre-emptive: urgency 6-7 — clear NOW, penalty is already 6^1.5×0.5=11.6/step
    #    Skip if current direction is already serving it
    if ns_em > 0 and ns_urg >= 6:
        if cur in (0, 3):
            return 0, f"preemptive-NS-hold(urg={ns_urg})"
        if ew_em == 0 or ns_urg >= ew_urg:
            return 0, f"preemptive-NS-switch(urg={ns_urg})"
    if ew_em > 0 and ew_urg >= 6:
        if cur in (1, 4):
            return 1, f"preemptive-EW-hold(urg={ew_urg})"
        if ns_em == 0 or ew_urg >= ns_urg:
            return 1, f"preemptive-EW-switch(urg={ew_urg})"

    # 4. Moderate emergency (urgency 5)
    if ns_em > 0 and ns_urg >= 5:
        if ew_em == 0 or ns_urg >= ew_urg:
            return 0, f"moderate-NS(urg={ns_urg})"
    if ew_em > 0 and ew_urg >= 5:
        return 1, f"moderate-EW(urg={ew_urg})"

    # 5. Hysteresis
    hold  = _dynamic_hold_time(obs, task)
    ratio = SWITCH_RATIO_BY_TASK.get(task, 1.3)
    if obs.time_in_phase < hold:
        if cur in (0, 3): return 0, f"hysteresis-NS(held={obs.time_in_phase}<{hold})"
        if cur in (1, 4): return 1, f"hysteresis-EW(held={obs.time_in_phase}<{hold})"

    # 6. Queue pressure
    ns_p, ew_p = _compute_pressures(obs)
    if ns_p > ew_p * ratio:
        return 0, f"pressure-NS({ns_p:.1f}>{ew_p:.1f}×{ratio})"
    if ew_p > ns_p * ratio:
        return 1, f"pressure-EW({ew_p:.1f}>{ns_p:.1f}×{ratio})"

    # 7. Hold current
    if cur in (0, 3): return 0, "hold-NS"
    if cur in (1, 4): return 1, "hold-EW"
    return 0, "default-NS"


def _should_skip_llm(obs: TrafficObservation, heuristic_phase: int, reason: str, step: int) -> bool:
    """
    Skip the LLM call when the decision is mathematically obvious OR outside the LLM cadence.
    LLM is only called every LLM_CALL_EVERY steps AND only for genuinely ambiguous pressure cases.
    """
    # Rate-limit: only consider LLM every N steps
    if step % LLM_CALL_EVERY != 0:
        return True
    # Always skip for time-critical or clear-cut decisions
    if "collision-risk" in reason:
        return True
    if "critical" in reason or "preemptive" in reason:
        return True
    if "hysteresis" in reason:
        return True
    if "hold" in reason:
        return True
    # Skip when pressure ratio is clear (> 2×) — heuristic is strictly better here
    ns_p, ew_p = _compute_pressures(obs)
    max_p = max(ns_p, ew_p, 0.01)
    min_p = min(ns_p, ew_p, 0.01)
    if max_p / min_p > 2.0:
        return True
    return False


# ---------------------------------------------------------------------------
# Live grade projection
# ---------------------------------------------------------------------------

def _project_score(task: str, state: Optional[TrafficState], step: int) -> str:
    if state is None or step == 0:
        return "(no data yet)"

    s     = state
    steps = max(s.step_count, 1)

    throughput_per_step = s.total_vehicles_passed / steps
    em_rate             = s.total_emergency_passed / steps
    avg_wait            = s.total_waiting_time / steps

    if task == "basic_flow":
        tput  = min(throughput_per_step / 1.8, 1.0)
        eff   = 1.0 / (1.0 + avg_wait * 0.1)
        sw    = s.total_phase_changes / steps
        stab  = max(0.0, 0.05 * (1.0 - min(sw * 4, 1.0)))
        proj  = tput * 0.6 + eff * 0.4 + stab
        gap   = max(0.0, 1.8 - throughput_per_step)
        return (
            f"projected={proj:.3f} | "
            f"throughput={tput:.2f}(×0.6, {throughput_per_step:.2f}veh/step, need +{gap:.2f}) "
            f"eff={eff:.2f}(×0.4) stab={stab:.3f}(switch={sw:.2f}/step, want<0.25)"
        )

    if task == "emergency_priority":
        tput     = min(throughput_per_step / 1.5, 1.0)
        em_score = min(em_rate / (1.0 / 20.0), 1.0)
        if s.total_emergency_passed > 0:
            delay = max(0.0, 1.0 - (s.total_emergency_delay / s.total_emergency_passed) / 12.0)
            avg_d = s.total_emergency_delay / s.total_emergency_passed
        else:
            delay = 0.5
            avg_d = float("inf")
        eff  = 1.0 / (1.0 + avg_wait * 0.05)
        proj = tput * 0.30 + em_score * 0.35 + delay * 0.20 + eff * 0.15
        return (
            f"projected={proj:.3f} | "
            f"em_rate={em_score:.2f}(×0.35, need 1em/20steps) "
            f"delay={delay:.2f}(×0.20, avg={avg_d:.1f}steps, want<3) "
            f"tput={tput:.2f}(×0.30) eff={eff:.2f}(×0.15)"
        )

    if task == "dynamic_scenarios":
        tput     = min(throughput_per_step / 2.0, 1.0)
        em_score = min(em_rate / (1.0 / 15.0), 1.0)
        if s.total_emergency_passed > 0:
            delay = max(0.0, 1.0 - (s.total_emergency_delay / s.total_emergency_passed) / 5.0)
            avg_d = s.total_emergency_delay / s.total_emergency_passed
        else:
            delay = 0.0
            avg_d = float("inf")
        eff   = 1.0 / (1.0 + avg_wait * 0.08)
        adapt = 1.0 / (1.0 + (s.total_phase_changes / steps) * 0.5)
        proj  = tput * 0.25 + em_score * 0.30 + delay * 0.20 + eff * 0.15 + adapt * 0.10
        return (
            f"projected={proj:.3f} | "
            f"em_rate={em_score:.2f}(×0.30) delay={delay:.2f}(×0.20,avg={avg_d:.1f}) "
            f"tput={tput:.2f}(×0.25) eff={eff:.2f}(×0.15) adapt={adapt:.2f}(×0.10)"
        )

    return "(unknown task)"


# ---------------------------------------------------------------------------
# System prompts (task-specific)
# ---------------------------------------------------------------------------

_SYSTEM_BASE = """You are an expert Autonomous Traffic Signal Controller for a 4-way intersection.

PHASES:  0=NS_GREEN  1=EW_GREEN  2=ALL_RED
FLOW RATES: NS_GREEN clears ~3 NS vehicles/step, EW_GREEN clears ~3 EW vehicles/step, ALL_RED clears 0.

REWARD PER STEP:
  +0.30 × regular vehicles cleared
  +12.0 × emergency vehicles cleared
  -(urgency^1.5)×0.5 per WAITING emergency vehicle (compounds EVERY step it waits!)
     urgency=5 → 5.59/step, urgency=7 → 9.26/step, urgency=8 → 11.31/step
  -0.08 × total vehicles waiting
  -0.5 to -2.0 for unnecessary phase switch (proportional to empty-queue ratio)
  +0.05 stability bonus when traffic flows without switching
  -200  for gridlock collision (episode ends immediately!)

SWITCHING COST vs BENEFIT:
  Never switch to an empty direction (full -2.0 penalty, zero gain).
  Each unnecessary switch also hurts stability/adaptability scores.
  A switch is justified ONLY when:
    (a) emergency vehicle in new direction, OR
    (b) new direction pressure > current direction pressure × task_ratio

TASK-SPECIFIC GRADING:"""

_SYSTEM_TASK_HINTS = {
    "basic_flow": """
  basic_flow weights: throughput×0.60  efficiency×0.40  stability_bonus
  TARGET: 1.8 vehicles/step throughput. Switch rate < 0.25/step for stability bonus.
  STRATEGY: Hold green phases 5-8 steps. Only switch when NS/EW queue imbalance > 50%.
  DO NOT switch to a direction with 0 vehicles — full penalty, zero reward.""",

    "emergency_priority": """
  emergency_priority weights: em_rate×0.35  throughput×0.30  delay×0.20  efficiency×0.15
  TARGET: Clear 1 emergency per 20 steps. Keep avg emergency delay < 3 steps.
  STRATEGY: Pre-clear any urgency≥6 direction IMMEDIATELY — at urgency=6, cost is 11.7/step.
  Emergency waiting one extra step costs more than 30 regular vehicles cleared.""",

    "dynamic_scenarios": """
  dynamic_scenarios weights: em_rate×0.30  throughput×0.25  delay×0.20  efficiency×0.15  adaptability×0.10
  TARGET: 2.0 vehicles/step throughput + clear all emergencies fast. Zero collisions.
  STRATEGY: Balance throughput and emergency response. Watch for surge traffic (queue growth > +3/step).
  Adaptability penalises OVER-switching — switch only when needed, not on impulse.""",
}

_SYSTEM_SUFFIX = """

DECISION RULES (strictly in order):
  1. Total queued > 28 AND held > 14 steps → rotate to larger queue (collision prevention!)
  2. Any urgency ≥ 8 emergency → switch to that direction IMMEDIATELY
  3. Any urgency ≥ 6 emergency in other direction → switch to clear before escalation
  4. Hold current phase until dynamic hold time (varies by queue depth)
  5. Switch only when other direction pressure > current × task_ratio
  6. ALL_RED ONLY when BOTH directions have critical emergencies simultaneously

Think step by step about (a) emergencies, (b) collision risk, (c) throughput/score impact.
Output ONLY valid JSON on the last line: {"light_phase": 0}"""


def _get_system_prompt(task: str) -> str:
    hint = _SYSTEM_TASK_HINTS.get(task, "")
    return _SYSTEM_BASE + hint + _SYSTEM_SUFFIX


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    obs: TrafficObservation,
    step: int,
    task: str,
    history: Deque[str],
    heuristic: int,
    heuristic_reason: str,
    score_projection: str,
) -> str:
    ns_p, ew_p = _compute_pressures(obs)
    ns_q   = obs.queue_lengths[0] + obs.queue_lengths[1]
    ew_q   = obs.queue_lengths[2] + obs.queue_lengths[3]
    ns_em  = obs.emergency_queue[0] + obs.emergency_queue[1]
    ew_em  = obs.emergency_queue[2] + obs.emergency_queue[3]
    ns_urg = max(obs.emergency_urgency[0], obs.emergency_urgency[1])
    ew_urg = max(obs.emergency_urgency[2], obs.emergency_urgency[3])
    total_q = sum(obs.queue_lengths)

    phase_name = {0: "NS_GREEN", 1: "EW_GREEN", 2: "ALL_RED", 3: "NS_YELLOW", 4: "EW_YELLOW"}
    hint_name  = {0: "NS_GREEN (0)", 1: "EW_GREEN (1)", 2: "ALL_RED (2)"}
    trend_str  = f"[{obs.queue_trend[0]:+d},{obs.queue_trend[1]:+d},{obs.queue_trend[2]:+d},{obs.queue_trend[3]:+d}]"

    # Emergency penalty cost — helps LLM quantify urgency
    ns_em_cost = f"{ns_em * (max(ns_urg,1)**1.5)*0.5:.1f}/step" if ns_em > 0 else "none"
    ew_em_cost = f"{ew_em * (max(ew_urg,1)**1.5)*0.5:.1f}/step" if ew_em > 0 else "none"

    collision_warn = ""
    if _collision_risk(obs):
        collision_warn = f"\n  ⚠ COLLISION RISK: {total_q} vehicles queued, held {obs.time_in_phase} steps!"

    history_str = "\n".join(history) if history else "  (episode start)"

    return (
        f"TASK: {task}  |  Step {step}\n"
        f"Phase: {phase_name.get(obs.current_phase, '?')} held {obs.time_in_phase} steps{collision_warn}\n"
        f"\n"
        f"CURRENT STATE:\n"
        f"  NS: {ns_q} regular + {ns_em} emergency(urgency={ns_urg}, cost={ns_em_cost})  pressure={ns_p:.1f}\n"
        f"  EW: {ew_q} regular + {ew_em} emergency(urgency={ew_urg}, cost={ew_em_cost})  pressure={ew_p:.1f}\n"
        f"  Total queued: {total_q}  Trend [N,S,E,W]: {trend_str}\n"
        f"  Avg wait: {obs.avg_wait_time:.1f} steps  |  Collision flag: {obs.collision}\n"
        f"\n"
        f"LIVE SCORE:\n  {score_projection}\n"
        f"\n"
        f"RECENT STEPS:\n{history_str}\n"
        f"\n"
        f"Heuristic says: {hint_name.get(heuristic, str(heuristic))}  ({heuristic_reason})\n"
        f"Reason through, then output JSON on the last line."
    )


# ---------------------------------------------------------------------------
# Parse LLM output
# ---------------------------------------------------------------------------

def _parse_phase(raw: str) -> Optional[int]:
    """Extract phase from LLM chain-of-thought output (JSON on last line)."""
    import re
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    for line in reversed(lines):
        try:
            data = json.loads(line)
            p = int(data.get("light_phase", data.get("phase", -1)))
            if 0 <= p <= 2:
                return p
        except Exception:
            pass
    m = re.search(r'"light_phase"\s*:\s*([012])', raw)
    if m:
        return int(m.group(1))
    m = re.search(r'\b([012])\b', raw)
    if m:
        return int(m.group(1))
    return None


def _sanitize(s: str) -> str:
    return s.replace('"', "'").replace("\n", " ")


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def get_action(
    client: OpenAI,
    obs: TrafficObservation,
    step: int,
    task: str,
    history: Deque[str],
    state: Optional[TrafficState],
    force_heuristic: bool,
) -> Tuple[TrafficAction, str]:
    """
    Returns (action, source) where source is "heuristic", "llm", or "fallback".
    Uses LLM only for ambiguous cases at the LLM cadence; pure heuristic otherwise.
    force_heuristic=True when task time budget is nearly exhausted.
    """
    heuristic, reason = _heuristic_phase(obs, task)

    # Fast path — skip LLM (obvious decision, wrong cadence, or budget exhausted)
    if force_heuristic or _should_skip_llm(obs, heuristic, reason, step):
        return TrafficAction(light_phase=heuristic), f"heuristic({reason})"

    # Ambiguous case — call LLM with a hard per-call timeout
    score_proj = _project_score(task, state, step)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _get_system_prompt(task)},
                {"role": "user",   "content": _build_prompt(
                    obs, step, task, history, heuristic, reason, score_proj
                )},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=LLM_TIMEOUT_S,
        )
        raw   = resp.choices[0].message.content.strip()
        phase = _parse_phase(raw)
        if phase is not None:
            return TrafficAction(light_phase=phase), "llm"
    except Exception:
        pass

    # Fallback to heuristic if LLM fails or times out
    return TrafficAction(light_phase=heuristic), f"fallback({reason})"


# ---------------------------------------------------------------------------
# Server health check
# ---------------------------------------------------------------------------

def _wait_for_server(url: str, timeout: int = 60) -> None:
    health_url = url.rstrip("/") + "/health"
    deadline   = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    print(f"[WARN] Server not healthy after {timeout}s — proceeding anyway", flush=True)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(task: str, client: OpenAI) -> dict:
    print(f'[START] task={task} env=traffic_control model={MODEL_NAME}', flush=True)

    rewards:         List[float]            = []
    history:         Deque[str]             = deque(maxlen=8)
    step             = 0
    last_error:      Optional[str]          = None
    done             = False
    state:           Optional[TrafficState] = None
    llm_calls        = 0
    heur_calls       = 0
    task_start_time  = time.time()
    budget_s         = TASK_BUDGET_S.get(task, 600)

    try:
        with TrafficControlEnv(base_url=SERVER_URL).sync() as env:
            reset_result = env.reset(task_id=task, seed=SEED)
            obs: TrafficObservation = reset_result.observation
            done = reset_result.done

            while not done:
                step += 1

                # Refresh cumulative state every 5 steps for score projection
                if step % 5 == 1:
                    try:
                        state = env.state()
                    except Exception:
                        pass

                # Switch to pure heuristic if we're within 60s of the task budget
                elapsed          = time.time() - task_start_time
                force_heuristic  = elapsed > budget_s - 60

                action, source = get_action(
                    client, obs, step, task, history, state,
                    force_heuristic,
                )
                action_str     = f"light_phase={action.light_phase}"

                if source.startswith("llm"):
                    llm_calls += 1
                else:
                    heur_calls += 1

                try:
                    result     = env.step(action)
                    obs        = result.observation
                    reward_val = result.reward if result.reward is not None else 0.0
                    rewards.append(reward_val)
                    done       = result.done
                    last_error = None

                    phase_name = {0: "NS", 1: "EW", 2: "AR", 3: "NSy", 4: "EWy"}
                    ns_urg = max(obs.emergency_urgency[0], obs.emergency_urgency[1])
                    ew_urg = max(obs.emergency_urgency[2], obs.emergency_urgency[3])
                    em_info = ""
                    if any(q > 0 for q in obs.emergency_queue):
                        em_info = f" EM[{obs.emergency_queue[0]+obs.emergency_queue[1]}u{ns_urg}|{obs.emergency_queue[2]+obs.emergency_queue[3]}u{ew_urg}]"
                    history.append(
                        f"  s{step}: {source[:4]}→{action.light_phase}"
                        f" clr={obs.vehicles_passed}r+{obs.emergency_passed}em"
                        f" r={reward_val:+.1f}"
                        f" ph={phase_name.get(obs.current_phase, '?')}"
                        f" q={list(obs.queue_lengths)}{em_info}"
                    )
                except Exception as exc:
                    reward_val = 0.0
                    done       = True
                    last_error = _sanitize(str(exc))

                error_str = "null" if last_error is None else f'"{last_error}"'
                print(
                    f'[STEP] step={step} action={action_str} '
                    f'reward={reward_val:.2f} done={str(done).lower()} error={error_str}',
                    flush=True,
                )

                if done:
                    break

    except Exception as exc:
        last_error = _sanitize(str(exc))
        print(f"[WARN] Episode error: {last_error}", flush=True)

    success      = done and last_error is None
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    total_reward = sum(rewards)
    max_possible = step * 10.0
    score        = min(0.999, max(0.001, total_reward / max_possible)) if max_possible > 0 else 0.001

    print(
        f'[END] success={str(success).lower()} steps={step} '
        f'score={score:.3f} rewards={rewards_str} '
        f'llm_calls={llm_calls} heuristic_calls={heur_calls}',
        flush=True,
    )

    return {"success": success, "steps": step, "rewards": rewards}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _wait_for_server(SERVER_URL)

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    tasks = ["basic_flow", "emergency_priority", "dynamic_scenarios"]
    for task in tasks:
        run_task(task, client)


if __name__ == "__main__":
    main()
