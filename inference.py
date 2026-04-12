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
MAX_TOKENS   = 256   # enough for chain-of-thought + JSON
TEMPERATURE  = 0.0

# Minimum green hold steps (ensures stability_bonus in grading)
MIN_GREEN_HOLD = 4

# Task max_steps for score projection
TASK_MAX_STEPS = {"basic_flow": 200, "emergency_priority": 300, "dynamic_scenarios": 400}


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


def _dynamic_hold_time(obs: TrafficObservation) -> int:
    """
    Adaptive minimum hold: longer when current direction has more traffic
    so we don't leave vehicles stranded mid-queue.
    """
    current = obs.current_phase
    if current in (0, 3):   # NS_GREEN / NS_YELLOW
        q = obs.queue_lengths[0] + obs.queue_lengths[1]
    elif current in (1, 4): # EW_GREEN / EW_YELLOW
        q = obs.queue_lengths[2] + obs.queue_lengths[3]
    else:
        return MIN_GREEN_HOLD
    # Hold longer if queue is deep (drain rate ~3 veh/step)
    return max(MIN_GREEN_HOLD, min(q // 3, 10))


def _heuristic_phase(obs: TrafficObservation) -> int:
    """
    Mathematically optimal phase recommendation.
    Priority order:
      1. Critical emergency (urgency ≥ 8) — clear NOW
      2. Moderate emergency (urgency 5-7) — prioritise unless other side is worse
      3. Hysteresis — don't switch if hold time not reached
      4. Queue pressure — switch to higher pressure direction
      5. Default — hold current
    """
    ns_em  = obs.emergency_queue[0] + obs.emergency_queue[1]
    ew_em  = obs.emergency_queue[2] + obs.emergency_queue[3]
    ns_urg = max(obs.emergency_urgency[0], obs.emergency_urgency[1])
    ew_urg = max(obs.emergency_urgency[2], obs.emergency_urgency[3])
    cur    = obs.current_phase

    # 1. Critical emergency
    if ns_em > 0 and ns_urg >= 8 and ew_em > 0 and ew_urg >= 8:
        return 2  # ALL_RED: both critical, momentary pause to avoid collision
    if ns_em > 0 and ns_urg >= 8:
        return 0
    if ew_em > 0 and ew_urg >= 8:
        return 1

    # 2. Moderate emergency — switch if other side isn't also urgent
    if ns_em > 0 and ns_urg >= 5:
        if ew_em == 0 or ns_urg >= ew_urg:
            return 0
    if ew_em > 0 and ew_urg >= 5:
        return 1

    # 3. Hysteresis
    hold = _dynamic_hold_time(obs)
    if obs.time_in_phase < hold:
        if cur in (0, 3): return 0
        if cur in (1, 4): return 1

    # 4. Queue pressure
    ns_p, ew_p = _compute_pressures(obs)
    if ns_p > ew_p * 1.3:
        return 0
    if ew_p > ns_p * 1.3:
        return 1

    # 5. Hold current
    if cur in (0, 3): return 0
    if cur in (1, 4): return 1
    return 0


# ---------------------------------------------------------------------------
# Live grade projection (shows LLM how its decisions impact the final score)
# ---------------------------------------------------------------------------

def _project_score(task: str, state: Optional[TrafficState], step: int) -> str:
    """Compute projected grading scores from current episode state."""
    if state is None or step == 0:
        return "(no data yet)"

    s     = state
    steps = max(s.step_count, 1)
    max_s = TASK_MAX_STEPS.get(task, 300)

    throughput_per_step = s.total_vehicles_passed / steps
    em_rate             = s.total_emergency_passed / steps
    avg_wait            = s.total_waiting_time / steps

    if task == "basic_flow":
        tput  = min(throughput_per_step / 1.8, 1.0)
        eff   = 1.0 / (1.0 + avg_wait * 0.1)
        sw    = s.total_phase_changes / steps
        stab  = max(0.0, 0.05 * (1.0 - min(sw * 4, 1.0)))
        proj  = tput * 0.6 + eff * 0.4 + stab
        return (
            f"throughput={tput:.2f}(×0.6) eff={eff:.2f}(×0.4) stability={stab:.3f} "
            f"→ projected={proj:.3f}  "
            f"[veh/step={throughput_per_step:.2f} target=1.8, switch_rate={sw:.2f} target<0.25]"
        )

    if task == "emergency_priority":
        tput     = min(throughput_per_step / 1.5, 1.0)
        em_score = min(em_rate / (1.0 / 20.0), 1.0)
        if s.total_emergency_passed > 0:
            delay = max(0.0, 1.0 - (s.total_emergency_delay / s.total_emergency_passed) / 12.0)
        else:
            delay = 0.5
        eff  = 1.0 / (1.0 + avg_wait * 0.05)
        proj = tput * 0.30 + em_score * 0.35 + delay * 0.20 + eff * 0.15
        return (
            f"tput={tput:.2f}(×0.30) em_rate={em_score:.2f}(×0.35) "
            f"delay={delay:.2f}(×0.20) eff={eff:.2f}(×0.15) → projected={proj:.3f}  "
            f"[em_cleared={s.total_emergency_passed} need≥{steps//20}]"
        )

    if task == "dynamic_scenarios":
        tput     = min(throughput_per_step / 2.0, 1.0)
        em_score = min(em_rate / (1.0 / 15.0), 1.0)
        if s.total_emergency_passed > 0:
            delay = max(0.0, 1.0 - (s.total_emergency_delay / s.total_emergency_passed) / 5.0)
        else:
            delay = 0.0
        eff   = 1.0 / (1.0 + avg_wait * 0.08)
        adapt = 1.0 / (1.0 + (s.total_phase_changes / steps) * 0.5)
        proj  = tput * 0.25 + em_score * 0.30 + delay * 0.20 + eff * 0.15 + adapt * 0.10
        return (
            f"tput={tput:.2f}(×0.25) em={em_score:.2f}(×0.30) delay={delay:.2f}(×0.20) "
            f"eff={eff:.2f}(×0.15) adapt={adapt:.2f}(×0.10) → projected={proj:.3f}"
        )

    return "(unknown task)"


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Autonomous Traffic Signal Controller optimising a 4-way intersection.

PHASES:  0=NS_GREEN  1=EW_GREEN  2=ALL_RED

REWARD FUNCTION (per step):
  +0.30 × regular vehicles cleared
  +12.0 × emergency vehicles cleared
  -(urgency^1.5)×0.5 per waiting emergency vehicle (EVERY step it waits!)
  -0.08 × total vehicles waiting
  -0.5 to -2.0 for unnecessary phase switch (proportional to how empty the new direction is)
  +0.05 stability bonus when traffic flows without switching

GRADING WEIGHTS:
  basic_flow:          throughput×0.60  efficiency×0.40  stability_bonus
  emergency_priority:  throughput×0.30  em_rate×0.35     delay×0.20  efficiency×0.15
  dynamic_scenarios:   throughput×0.25  em_rate×0.30     delay×0.20  efficiency×0.15  adaptability×0.10

DECISION RULES (follow strictly):
  1. Urgency ≥ 8 emergency → switch to that direction IMMEDIATELY
  2. Urgency 5-7 emergency → switch unless other side is equally urgent
  3. Hold current phase ≥ 4 steps before switching (preserves stability bonus)
  4. Switch only when pressure ratio > 1.3× (avoid unnecessary switches)
  5. ALL_RED only when both directions have simultaneous critical emergencies

Think step by step, then output ONLY valid JSON on the last line: {"light_phase": 0}"""


def _build_prompt(
    obs: TrafficObservation,
    step: int,
    task: str,
    history: Deque[str],
    heuristic: int,
    score_projection: str,
) -> str:
    ns_p, ew_p = _compute_pressures(obs)
    ns_q  = obs.queue_lengths[0] + obs.queue_lengths[1]
    ew_q  = obs.queue_lengths[2] + obs.queue_lengths[3]
    ns_em = obs.emergency_queue[0] + obs.emergency_queue[1]
    ew_em = obs.emergency_queue[2] + obs.emergency_queue[3]
    ns_urg = max(obs.emergency_urgency[0], obs.emergency_urgency[1])
    ew_urg = max(obs.emergency_urgency[2], obs.emergency_urgency[3])

    phase_name = {0:"NS_GREEN",1:"EW_GREEN",2:"ALL_RED",3:"NS_YELLOW",4:"EW_YELLOW"}
    hint_name  = {0:"NS_GREEN (0)",1:"EW_GREEN (1)",2:"ALL_RED (2)"}
    trend_str  = f"[{obs.queue_trend[0]:+d},{obs.queue_trend[1]:+d},{obs.queue_trend[2]:+d},{obs.queue_trend[3]:+d}]"

    history_str = "\n".join(history) if history else "  (episode start)"

    return (
        f"TASK: {task}  |  Step {step}  |  Current phase: {phase_name.get(obs.current_phase,'?')} (held {obs.time_in_phase} steps)\n"
        f"\n"
        f"STATE:\n"
        f"  NS: {ns_q} regular + {ns_em} emergency(urgency={ns_urg})  pressure={ns_p:.1f}\n"
        f"  EW: {ew_q} regular + {ew_em} emergency(urgency={ew_urg})  pressure={ew_p:.1f}\n"
        f"  Queues [N,S,E,W]: {list(obs.queue_lengths)}  trend={trend_str}\n"
        f"  Avg wait: {obs.avg_wait_time:.1f} steps  |  Collision: {obs.collision}\n"
        f"\n"
        f"LIVE SCORE PROJECTION:\n  {score_projection}\n"
        f"\n"
        f"RECENT HISTORY (last {len(history)} steps):\n{history_str}\n"
        f"\n"
        f"Heuristic recommendation: {hint_name[heuristic]}\n"
        f"Reason through the decision, then output JSON on the last line."
    )


# ---------------------------------------------------------------------------
# Parse + LLM call
# ---------------------------------------------------------------------------

def _sanitize(s: str) -> str:
    return s.replace('"', "'").replace("\n", " ")


def _parse_phase(raw: str) -> Optional[int]:
    """Extract phase from LLM chain-of-thought output (JSON on last line)."""
    import re
    # Try last non-empty line first (chain-of-thought ends with JSON)
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    for line in reversed(lines):
        try:
            data = json.loads(line)
            p = int(data.get("light_phase", data.get("phase", -1)))
            if 0 <= p <= 2:
                return p
        except Exception:
            pass
    # Fallback: regex anywhere
    m = re.search(r'"light_phase"\s*:\s*([012])', raw)
    if m:
        return int(m.group(1))
    m = re.search(r'\b([012])\b', raw)
    if m:
        return int(m.group(1))
    return None


def get_llm_action(
    client: OpenAI,
    obs: TrafficObservation,
    step: int,
    task: str,
    history: Deque[str],
    state: Optional[TrafficState],
) -> TrafficAction:
    heuristic  = _heuristic_phase(obs)
    score_proj = _project_score(task, state, step)

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_prompt(obs, step, task, history, heuristic, score_proj)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    raw   = resp.choices[0].message.content.strip()
    phase = _parse_phase(raw)

    # Fallback to heuristic if LLM output is unparseable or clearly wrong
    if phase is None:
        phase = heuristic

    return TrafficAction(light_phase=phase)


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

    rewards:    List[float]        = []
    history:    Deque[str]         = deque(maxlen=6)
    step        = 0
    last_error: Optional[str]      = None
    done        = False
    state:      Optional[TrafficState] = None

    try:
        with TrafficControlEnv(base_url=SERVER_URL).sync() as env:
            reset_result = env.reset(task_id=task, seed=SEED)
            obs: TrafficObservation = reset_result.observation
            done = reset_result.done

            while not done:
                step += 1

                # Refresh cumulative state every 10 steps for score projection
                if step % 10 == 1:
                    try:
                        state = env.state()
                    except Exception:
                        pass

                action     = get_llm_action(client, obs, step, task, history, state)
                action_str = f"light_phase={action.light_phase}"

                try:
                    result     = env.step(action)
                    obs        = result.observation
                    reward_val = result.reward if result.reward is not None else 0.0
                    rewards.append(reward_val)
                    done       = result.done
                    last_error = None

                    phase_name = {0:"NS",1:"EW",2:"AR",3:"NSy",4:"EWy"}
                    history.append(
                        f"  s{step}: →{action.light_phase}"
                        f" clr={obs.vehicles_passed}r+{obs.emergency_passed}em"
                        f" r={reward_val:+.1f}"
                        f" now={phase_name.get(obs.current_phase,'?')}"
                        f" queues={list(obs.queue_lengths)}"
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
        f'score={score:.3f} rewards={rewards_str}',
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
