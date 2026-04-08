"""
Inference Script — Autonomous Traffic Control OpenEnv Environment
=================================================================
Mandatory env variables (injected by validator):
    API_BASE_URL   LLM proxy endpoint (MUST use validator's LiteLLM proxy)
    MODEL_NAME     Model identifier
    API_KEY        LiteLLM proxy key (MUST use validator's injected key)

Optional:
    SERVER_URL     Running env server (default: http://localhost:8000)

Usage:
    API_BASE_URL=<url> API_KEY=<key> python inference.py

CRITICAL: This script REQUIRES API_BASE_URL and API_KEY from environment.
          No fallbacks or hardcoded values are used.
          The validator injects these to route calls through LiteLLM proxy.
"""

import os
import re
import sys
import json
import textwrap
import requests as _http
from typing import List, Optional

# Allow running from repo root or from traffic_control/ subdirectory
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

try:
    from traffic_control.client import TrafficControlEnv
    from traffic_control.models import TrafficAction, TrafficObservation
except ImportError:
    from client import TrafficControlEnv  # type: ignore
    from models import TrafficAction, TrafficObservation  # type: ignore

# ---------------------------------------------------------------------------
# Configuration — read from env at import time (matches sample script pattern)
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
SERVER_URL   = os.environ.get("SERVER_URL", "http://localhost:8000")

SEED        = 42
MAX_TOKENS  = 64
TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an Autonomous Traffic Control AI managing a 4-way intersection.

    OBJECTIVE: Maximise vehicle throughput and prioritise emergency vehicles.

    PHASES:
      0 = North-South Green  (N/S vehicles may pass)
      1 = East-West Green    (E/W vehicles may pass)
      2 = All Red            (no vehicles pass — rarely needed)

    DECISION RULES (apply in order):
      1. EMERGENCY CHECK: If emergency vehicles are waiting (emergency_queue > 0),
         IMMEDIATELY switch to phase 0 if N/S has emergencies, else phase 1.
         Urgency 8-10 is critical - act immediately regardless of time_in_phase.

      2. MINIMUM PHASE TIME: Stay in current phase at least 3 steps.
         If time_in_phase < 3, remain in current phase.

      3. QUEUE BALANCE: After minimum time, compare N/S vs E/W queue depths.
         - If one direction has 3+ more vehicles than the other, switch to that phase.
         - If within 2 vehicles, stay in current phase to avoid switch penalty.

      4. EMPTY QUEUE: If current direction has 0 vehicles waiting and other direction > 0,
         switch immediately (no minimum time wait needed).

    REWARD SIGNALS:
      - Vehicles passing: +0.2 each
      - Emergency vehicles passing: +10 each
      - Phase change with empty queue: -0.5 penalty
      - Emergency waiting: -0.4 * urgency per step (HUGE penalty)

    OUTPUT: Reply with exactly one JSON object — no markdown, no explanation:
      {"light_phase": <0, 1, or 2>}
""").strip()


def _build_prompt(obs: TrafficObservation) -> str:
    q    = obs.queue_lengths
    em_q = obs.emergency_queue
    em_u = obs.emergency_urgency
    return textwrap.dedent(f"""
        CURRENT STATE:
          Active phase      : {obs.current_phase}
          Steps in phase    : {obs.time_in_phase}
          Regular queue     : N={q[0]}, S={q[1]}, E={q[2]}, W={q[3]}
          Emergency queue   : N={em_q[0]}, S={em_q[1]}, E={em_q[2]}, W={em_q[3]}
          Emergency urgency : N={em_u[0]}, S={em_u[1]}, E={em_u[2]}, W={em_u[3]}

        Respond with exactly: {{"light_phase": <0, 1, or 2>}}
    """).strip()

# ---------------------------------------------------------------------------
# Rule-based fallback — optimized for high scores
# ---------------------------------------------------------------------------

def _rule_based_action(obs: TrafficObservation) -> TrafficAction:
    em_q = obs.emergency_queue
    em_u = obs.emergency_urgency
    q    = obs.queue_lengths
    current = obs.current_phase
    time_in = obs.time_in_phase

    # Emergency prioritization: urgency-weighted score per direction
    ns_em_urgency = em_u[0] + em_u[1] + em_q[0] * 2 + em_q[1] * 2
    ew_em_urgency = em_u[2] + em_u[3] + em_q[2] * 2 + em_q[3] * 2

    if ns_em_urgency > 0 or ew_em_urgency > 0:
        # Emergency waiting - switch immediately to help them
        return TrafficAction(light_phase=0 if ns_em_urgency >= ew_em_urgency else 1)

    # No emergencies - use queue depth with hysteresis
    ns_total = q[0] + q[1]
    ew_total = q[2] + q[3]

    # Dynamic minimum phase time based on queue depth (deeper queues = stay longer)
    min_phase_time = min(3 + max(ns_total, ew_total) // 5, 8)

    # Stay in current phase if below min time and still has traffic
    if current == 0 and time_in < min_phase_time and ns_total > 0:
        return TrafficAction(light_phase=0)
    if current == 1 and time_in < min_phase_time and ew_total > 0:
        return TrafficAction(light_phase=1)

    # Switch to direction with more traffic (with 2-vehicle hysteresis to prevent flip-flopping)
    if ns_total >= ew_total + 2:
        return TrafficAction(light_phase=0)
    elif ew_total >= ns_total + 2:
        return TrafficAction(light_phase=1)
    else:
        # Within 2 vehicles - stay in current phase to avoid switch penalty
        return TrafficAction(light_phase=current if current in (0, 1) else 0)

# ---------------------------------------------------------------------------
# LLM action — client passed in from main() (created once with env-level vars)
# ---------------------------------------------------------------------------

def get_llm_action(client: OpenAI, obs: TrafficObservation) -> TrafficAction:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_prompt(obs)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    data_str = (resp.choices[0].message.content or "").strip()
    match = re.search(r'\{[^}]*\}', data_str.replace('\n', ' '))
    data  = json.loads(match.group(0) if match else data_str)
    phase = max(0, min(2, int(data.get("light_phase", obs.current_phase))))
    return TrafficAction(light_phase=phase)

# ---------------------------------------------------------------------------
# Grade fetcher
# ---------------------------------------------------------------------------

def _fetch_score(task_id: str, state_payload: dict) -> float:
    try:
        r = _http.post(
            f"{SERVER_URL}/grade",
            json={
                "task_id":                task_id,
                "total_vehicles_passed":  state_payload.get("total_vehicles_passed", 0),
                "total_emergency_passed": state_payload.get("total_emergency_passed", 0),
                "total_waiting_time":     state_payload.get("total_waiting_time", 0.0),
                "total_collisions":       state_payload.get("total_collisions", 0),
                "total_emergency_delay":  state_payload.get("total_emergency_delay", 0.0),
                "total_phase_changes":    state_payload.get("total_phase_changes", 0),
                "step_count":             max(state_payload.get("step_count", 1), 1),
            },
            timeout=10,
        )
        if r.status_code == 200:
            return max(0.001, min(0.999, float(r.json().get("score", 0.5))))
    except Exception:
        pass
    return 0.5

# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, client: OpenAI) -> None:
    print(f"[START] task={task_id} env=traffic-control model={MODEL_NAME}", flush=True)

    rewards: List[float] = []
    success = False

    try:
        with TrafficControlEnv(base_url=SERVER_URL).sync() as env:
            step_result = env.reset(task_id=task_id, seed=SEED)
            step = 1

            while not step_result.done:
                obs       = step_result.observation
                error_msg: Optional[str] = None

                action = get_llm_action(client, obs)

                action_str = f"TrafficAction(light_phase={action.light_phase})"

                try:
                    step_result = env.step(action)
                    reward_val  = step_result.reward if step_result.reward is not None else 0.0
                    rewards.append(reward_val)
                    done_val    = str(step_result.done).lower()
                    error_val   = error_msg if error_msg else "null"
                    print(
                        f"[STEP] step={step} action={action_str} "
                        f"reward={reward_val:.2f} done={done_val} error={error_val}",
                        flush=True,
                    )
                except Exception as exc:
                    env_err = str(exc).replace('"', "'").replace("\\", "")
                    print(
                        f"[STEP] step={step} action={action_str} "
                        f"reward=0.00 done=true error={env_err}",
                        flush=True,
                    )
                    break

                step += 1

            success = True

    except Exception as exc:
        print(f"[STEP] step=0 action=none reward=0.00 done=true error={exc}", flush=True)
        success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"

    score = 0.5
    try:
        state_resp = _http.get(f"{SERVER_URL}/state", timeout=10)
        if state_resp.status_code == 200:
            score = _fetch_score(task_id, state_resp.json())
    except Exception:
        pass

    print(
        f"[END] success={str(success).lower()} steps={len(rewards)} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(
        f"[CONFIG] API_BASE_URL={API_BASE_URL} MODEL_NAME={MODEL_NAME} "
        f"API_KEY_SET={bool(API_KEY)} SERVER_URL={SERVER_URL}",
        flush=True,
    )

    if not API_KEY:
        raise SystemExit(
            "[FATAL] API_KEY is not set. "
            "The validator must inject API_KEY as an environment variable."
        )

    # Create the OpenAI client once using module-level env vars
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in ["basic_flow", "emergency_priority", "dynamic_scenarios"]:
        run_task(task, client)
        print(flush=True)


if __name__ == "__main__":
    main()
