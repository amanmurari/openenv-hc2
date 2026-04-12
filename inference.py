"""Inference Script — Autonomous Traffic Control OpenEnv Environment"""

import os
import sys
import json
import time
import urllib.request
from collections import deque
from typing import Deque, List, Optional, Tuple

_HERE   = os.path.dirname(os.path.abspath(__file__))
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
# Config — injected by hackathon validator
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
SERVER_URL   = os.getenv("SERVER_URL", "http://localhost:7860")
SEED         = 42
MAX_TOKENS   = 120    # short CoT + JSON — keeps latency low
TEMPERATURE  = 0.0
LLM_TIMEOUT  = 6     # per-call timeout in seconds

# Per-task wall-clock budget. Once elapsed > budget - 60s, force pure heuristic
# so the episode always finishes inside the budget.
# Total budget: 330+480+630 = 1440s = 24 min — well under the 30-min kill limit.
TASK_BUDGET_S = {
    "basic_flow":          330,
    "emergency_priority":  480,
    "dynamic_scenarios":   630,
}

TASK_MAX_STEPS = {"basic_flow": 200, "emergency_priority": 300, "dynamic_scenarios": 400}


# ---------------------------------------------------------------------------
# Heuristic (fallback only — used when budget is nearly exhausted or LLM fails)
# ---------------------------------------------------------------------------

def _em_pressure(count: int, urgency: int) -> float:
    return count * (max(urgency, 1) ** 1.5) * 0.5 if count > 0 else 0.0

def _dir_pressure(queue: int, em_count: int, urgency: int) -> float:
    return queue * 0.30 + _em_pressure(em_count, urgency) * 4.0

def _compute_pressures(obs: TrafficObservation) -> Tuple[float, float]:
    ns = _dir_pressure(
        obs.queue_lengths[0] + obs.queue_lengths[1],
        obs.emergency_queue[0] + obs.emergency_queue[1],
        max(obs.emergency_urgency[0], obs.emergency_urgency[1]),
    )
    ew = _dir_pressure(
        obs.queue_lengths[2] + obs.queue_lengths[3],
        obs.emergency_queue[2] + obs.emergency_queue[3],
        max(obs.emergency_urgency[2], obs.emergency_urgency[3]),
    )
    return ns, ew

def _heuristic_phase(obs: TrafficObservation, task: str) -> int:
    ns_em  = obs.emergency_queue[0] + obs.emergency_queue[1]
    ew_em  = obs.emergency_queue[2] + obs.emergency_queue[3]
    ns_urg = max(obs.emergency_urgency[0], obs.emergency_urgency[1])
    ew_urg = max(obs.emergency_urgency[2], obs.emergency_urgency[3])
    cur    = obs.current_phase

    # Critical emergency
    if ns_em > 0 and ns_urg >= 8 and ew_em > 0 and ew_urg >= 8:
        return 2
    if ns_em > 0 and ns_urg >= 8:
        return 0
    if ew_em > 0 and ew_urg >= 8:
        return 1

    # Moderate emergency
    if ns_em > 0 and ns_urg >= 5:
        if ew_em == 0 or ns_urg >= ew_urg:
            return 0
    if ew_em > 0 and ew_urg >= 5:
        return 1

    # Hysteresis
    min_hold = 6 if task == "basic_flow" else 3
    if obs.time_in_phase < min_hold:
        if cur in (0, 3): return 0
        if cur in (1, 4): return 1

    # Pressure
    ns_p, ew_p = _compute_pressures(obs)
    ratio = 1.5 if task == "basic_flow" else 1.2
    if ns_p > ew_p * ratio: return 0
    if ew_p > ns_p * ratio: return 1

    # Hold current
    if cur in (0, 3): return 0
    if cur in (1, 4): return 1
    return 0


# ---------------------------------------------------------------------------
# Live score projection
# ---------------------------------------------------------------------------

def _project_score(task: str, state: Optional[TrafficState], step: int) -> str:
    if state is None or step == 0:
        return "(no data yet)"
    s     = state
    steps = max(s.step_count, 1)
    tps   = s.total_vehicles_passed / steps
    er    = s.total_emergency_passed / steps
    aw    = s.total_waiting_time / steps

    if task == "basic_flow":
        tput = min(tps / 1.8, 1.0)
        eff  = 1.0 / (1.0 + aw * 0.1)
        sw   = s.total_phase_changes / steps
        stab = max(0.0, 0.05 * (1.0 - min(sw * 4, 1.0)))
        proj = tput * 0.6 + eff * 0.4 + stab
        return f"projected={proj:.3f} tput={tput:.2f}(×0.6,{tps:.2f}v/s,need1.8) eff={eff:.2f}(×0.4) stab={stab:.3f}(sw={sw:.2f}/s)"

    if task == "emergency_priority":
        tput = min(tps / 1.5, 1.0)
        ems  = min(er / (1.0/20.0), 1.0)
        if s.total_emergency_passed > 0:
            d    = max(0.0, 1.0 - (s.total_emergency_delay / s.total_emergency_passed) / 12.0)
            avgd = s.total_emergency_delay / s.total_emergency_passed
        else:
            d, avgd = 0.5, float("inf")
        eff  = 1.0 / (1.0 + aw * 0.05)
        proj = tput*0.30 + ems*0.35 + d*0.20 + eff*0.15
        return f"projected={proj:.3f} em={ems:.2f}(×0.35) delay={d:.2f}(×0.20,avg={avgd:.1f}) tput={tput:.2f}(×0.30)"

    if task == "dynamic_scenarios":
        tput = min(tps / 2.0, 1.0)
        ems  = min(er / (1.0/15.0), 1.0)
        if s.total_emergency_passed > 0:
            d    = max(0.0, 1.0 - (s.total_emergency_delay / s.total_emergency_passed) / 5.0)
            avgd = s.total_emergency_delay / s.total_emergency_passed
        else:
            d, avgd = 0.0, float("inf")
        eff  = 1.0 / (1.0 + aw * 0.08)
        ada  = 1.0 / (1.0 + (s.total_phase_changes / steps) * 0.5)
        proj = tput*0.25 + ems*0.30 + d*0.20 + eff*0.15 + ada*0.10
        return f"projected={proj:.3f} em={ems:.2f}(×0.30) delay={d:.2f}(avg={avgd:.1f}) tput={tput:.2f} eff={eff:.2f} ada={ada:.2f}"

    return "(unknown task)"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Autonomous Traffic Signal Controller for a 4-way intersection.

PHASES: 0=NS_GREEN (North-South green)  1=EW_GREEN (East-West green)  2=ALL_RED (emergency clearance)
FLOW: each GREEN phase clears ~3 vehicles/step in that direction. ALL_RED clears 0.

REWARD PER STEP:
  +0.30 × regular vehicles cleared
  +12.0 × emergency vehicles cleared
  -(urgency^1.5)×0.5 per WAITING emergency (every step it waits — urgency=6→7.4/step, urgency=8→11.3/step)
  -0.08 × total vehicles waiting
  -0.5 to -2.0 unnecessary phase switch (proportional to empty-queue ratio)
  +0.05 stability bonus (no switch this step)
  -200  gridlock collision (instant episode end!)

GRADING WEIGHTS:
  basic_flow:          throughput×0.60  efficiency×0.40  (+stability bonus)
  emergency_priority:  em_rate×0.35     throughput×0.30  delay×0.20  efficiency×0.15
  dynamic_scenarios:   em_rate×0.30     throughput×0.25  delay×0.20  efficiency×0.15  adaptability×0.10

RULES:
  1. Any urgency≥8 emergency → switch to that direction NOW (cost=11.3/step if you wait)
  2. Any urgency≥6 emergency → clear before it escalates (cost=7.4/step)
  3. basic_flow: hold each phase ≥6 steps; switch only when other direction queue is 50%+ bigger
  4. emergency tasks: react fast, hold ≥3 steps minimum
  5. ALL_RED only when BOTH directions have simultaneous critical emergencies
  6. Never switch to an empty direction (full -2.0 penalty, zero gain)
  7. Total queue >28 + held >14 steps → rotate to prevent -200 gridlock collision

Think step-by-step then output ONLY valid JSON on the last line: {"light_phase": 0}"""


def _build_prompt(
    obs: TrafficObservation,
    step: int,
    task: str,
    history: Deque[str],
    heuristic: int,
    score_proj: str,
) -> str:
    ns_p, ew_p = _compute_pressures(obs)
    ns_q   = obs.queue_lengths[0] + obs.queue_lengths[1]
    ew_q   = obs.queue_lengths[2] + obs.queue_lengths[3]
    ns_em  = obs.emergency_queue[0] + obs.emergency_queue[1]
    ew_em  = obs.emergency_queue[2] + obs.emergency_queue[3]
    ns_urg = max(obs.emergency_urgency[0], obs.emergency_urgency[1])
    ew_urg = max(obs.emergency_urgency[2], obs.emergency_urgency[3])
    total_q = sum(obs.queue_lengths)

    pname   = {0:"NS_GREEN",1:"EW_GREEN",2:"ALL_RED",3:"NS_YELLOW",4:"EW_YELLOW"}
    hname   = {0:"NS_GREEN(0)",1:"EW_GREEN(1)",2:"ALL_RED(2)"}
    trend   = f"[{obs.queue_trend[0]:+d},{obs.queue_trend[1]:+d},{obs.queue_trend[2]:+d},{obs.queue_trend[3]:+d}]"

    ns_cost = f"{ns_em*(max(ns_urg,1)**1.5)*0.5:.1f}/step" if ns_em > 0 else "none"
    ew_cost = f"{ew_em*(max(ew_urg,1)**1.5)*0.5:.1f}/step" if ew_em > 0 else "none"
    collision_warn = f"\n  *** COLLISION RISK: {total_q} queued, held {obs.time_in_phase} steps! ***" if total_q > 28 and obs.time_in_phase > 14 else ""

    hist_str = "\n".join(history) if history else "  (start)"

    return (
        f"TASK: {task}  Step {step}\n"
        f"Phase: {pname.get(obs.current_phase,'?')} held {obs.time_in_phase} steps{collision_warn}\n"
        f"\n"
        f"STATE:\n"
        f"  NS: {ns_q} vehicles + {ns_em} emergency(urgency={ns_urg}, cost={ns_cost})  pressure={ns_p:.1f}\n"
        f"  EW: {ew_q} vehicles + {ew_em} emergency(urgency={ew_urg}, cost={ew_cost})  pressure={ew_p:.1f}\n"
        f"  Total queued: {total_q}  Trend[N,S,E,W]: {trend}  Avg wait: {obs.avg_wait_time:.1f}s\n"
        f"\n"
        f"SCORE: {score_proj}\n"
        f"\n"
        f"HISTORY:\n{hist_str}\n"
        f"\n"
        f"Heuristic recommends: {hname.get(heuristic, str(heuristic))}\n"
        f"Reason through the decision, then output JSON on the last line."
    )


# ---------------------------------------------------------------------------
# Parse LLM output
# ---------------------------------------------------------------------------

def _parse_phase(raw: str) -> Optional[int]:
    import re
    for line in reversed([l.strip() for l in raw.split("\n") if l.strip()]):
        try:
            data = json.loads(line)
            p = int(data.get("light_phase", data.get("phase", -1)))
            if 0 <= p <= 2:
                return p
        except Exception:
            pass
    m = re.search(r'"light_phase"\s*:\s*([012])', raw)
    if m: return int(m.group(1))
    m = re.search(r'\b([012])\b', raw)
    if m: return int(m.group(1))
    return None


def _sanitize(s: str) -> str:
    return s.replace('"', "'").replace("\n", " ")


# ---------------------------------------------------------------------------
# Action: LLM every step, heuristic fallback
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
    heuristic = _heuristic_phase(obs, task)

    # Pure heuristic when time budget is nearly exhausted
    if force_heuristic:
        return TrafficAction(light_phase=heuristic), "heuristic(budget)"

    score_proj = _project_score(task, state, step)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_prompt(obs, step, task, history, heuristic, score_proj)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=LLM_TIMEOUT,
        )
        raw   = resp.choices[0].message.content.strip()
        phase = _parse_phase(raw)
        if phase is not None:
            return TrafficAction(light_phase=phase), "llm"
    except Exception:
        pass

    return TrafficAction(light_phase=heuristic), "heuristic(fallback)"


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

    rewards:        List[float]            = []
    history:        Deque[str]             = deque(maxlen=6)
    step            = 0
    last_error:     Optional[str]          = None
    done            = False
    state:          Optional[TrafficState] = None
    llm_calls       = 0
    task_start      = time.time()
    budget_s        = TASK_BUDGET_S.get(task, 600)

    try:
        with TrafficControlEnv(base_url=SERVER_URL).sync() as env:
            reset_result = env.reset(task_id=task, seed=SEED)
            obs: TrafficObservation = reset_result.observation
            done = reset_result.done

            while not done:
                step += 1

                # Refresh cumulative state every 10 steps
                if step % 10 == 1:
                    try:
                        state = env.state()
                    except Exception:
                        pass

                elapsed         = time.time() - task_start
                force_heuristic = elapsed > budget_s - 60

                action, source = get_action(
                    client, obs, step, task, history, state, force_heuristic
                )
                action_str = f"light_phase={action.light_phase}"
                if source == "llm":
                    llm_calls += 1

                try:
                    result     = env.step(action)
                    obs        = result.observation
                    reward_val = result.reward if result.reward is not None else 0.0
                    rewards.append(reward_val)
                    done       = result.done
                    last_error = None

                    pname  = {0:"NS",1:"EW",2:"AR",3:"NSy",4:"EWy"}
                    ns_urg = max(obs.emergency_urgency[0], obs.emergency_urgency[1])
                    ew_urg = max(obs.emergency_urgency[2], obs.emergency_urgency[3])
                    em_str = ""
                    if any(q > 0 for q in obs.emergency_queue):
                        ns_e = obs.emergency_queue[0]+obs.emergency_queue[1]
                        ew_e = obs.emergency_queue[2]+obs.emergency_queue[3]
                        em_str = f" EM[{ns_e}u{ns_urg}|{ew_e}u{ew_urg}]"
                    history.append(
                        f"  s{step}({source[:3]}):→{action.light_phase}"
                        f" clr={obs.vehicles_passed}r+{obs.emergency_passed}em"
                        f" r={reward_val:+.1f}"
                        f" ph={pname.get(obs.current_phase,'?')}"
                        f" q={list(obs.queue_lengths)}{em_str}"
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

    elapsed_total = time.time() - task_start
    print(
        f'[END] success={str(success).lower()} steps={step} '
        f'score={score:.3f} rewards={rewards_str} '
        f'llm_calls={llm_calls} elapsed={elapsed_total:.0f}s',
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

    for task in ["basic_flow", "emergency_priority", "dynamic_scenarios"]:
        run_task(task, client)


if __name__ == "__main__":
    main()
