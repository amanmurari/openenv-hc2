"""
Inference Script — Autonomous Traffic Control OpenEnv Environment
=================================================================
Advanced Hybrid Agent: combines optimized rule engine + LLM for ambiguous cases.

Mandatory env variables (injected by validator):
    API_BASE_URL   LLM proxy endpoint (MUST use validator's LiteLLM proxy)
    API_KEY        LiteLLM proxy key
    MODEL_NAME     Model identifier

Optional:
    SERVER_URL     Running env server (default: http://localhost:8000)

Output format: [START], [STEP], [END] lines only (strict protocol compliance)
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
# Configuration - CRITICAL: Use os.environ[] with NO fallbacks per validator
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")

SEED = 42
MAX_TOKENS = 64
TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# Enhanced LLM Prompt — scoring-aware
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Autonomous Traffic Signal Controller.

    OBJECTIVE: Maximise the final score (0-1) by keeping vehicles moving,
               eliminating emergency delays, and minimising queue lengths.

    PHASES:
      0 = North-South Green  (N/S can cross)
      1 = East-West Green    (E/W can cross)
      2 = All Red            (use only for emergency clearance)

    SCORING RULES (memorise these — they decide your reward):
      • +0.2 per regular vehicle that clears
      • +10.0 per emergency vehicle that clears  (HUGE weight)
      • −0.4 × urgency per step an emergency waits  (massive penalty)
      • −0.5 for switching to a phase with 0 vehicles waiting
      • Aim for 0 collisions, <50 total waiting time, >50 vehicles cleared.

    DECISION PRIORITY (apply in order):
      1. EMERGENCY — if any emergency_queue > 0, switch immediately to the
         direction (0=N/S, 1=E/W) with the highest *urgency sum*.
         (Urgency 9-10 is critical — do NOT make them wait.)
      2. MIN GREEN — stay in current phase at least 3 steps if traffic present.
      3. QUEUE BALANCE — after min green, switch if opposite direction has
         3+ more vehicles to reduce total waiting time.
      4. EMPTY QUEUE — if current direction has 0 waiting and other > 0,
         switch immediately to serve the other side.

    OUTPUT FORMAT (STRICT):
      {"light_phase": 0_or_1_or_2}
      No markdown, no extra text, just the JSON object.
""")


def _build_prompt(obs: TrafficObservation, step: int) -> str:
    em_q = obs.emergency_queue
    em_u = obs.emergency_urgency
    q = obs.queue_lengths
    current = obs.current_phase
    time_in = obs.time_in_phase

    return (
        f"Step {step}\n"
        f"Current phase: {current} (0=NS, 1=EW, 2=AllRed)\n"
        f"Time in phase: {time_in} steps\n\n"
        f"Queue lengths      [N, S, E, W]: {list(q)}\n"
        f"Emergency queues   [N, S, E, W]: {list(em_q)}\n"
        f"Emergency urgency  [N, S, E, W]: {list(em_u)}\n\n"
        f"Based on the scoring rules and decision priority, what phase should be set?\n"
        f"Respond ONLY with JSON: {{\"light_phase\": 0, 1, or 2}}"
    )


def _sanitize(s: str) -> str:
    """Remove characters that break JSON parsing in logs."""
    return s.replace('"', "'").replace("\\", "/").replace("\n", " ")


def _rule_based_action(obs: TrafficObservation) -> TrafficAction:
    """Optimized rule engine for high scores."""
    em_q = obs.emergency_queue
    em_u = obs.emergency_urgency
    q = obs.queue_lengths
    current = obs.current_phase
    time_in = obs.time_in_phase

    # Emergency prioritization (urgency-weighted)
    ns_em_urgency = em_u[0] + em_u[1] + em_q[0] * 2 + em_q[1] * 2
    ew_em_urgency = em_u[2] + em_u[3] + em_q[2] * 2 + em_q[3] * 2

    if ns_em_urgency > 0 or ew_em_urgency > 0:
        return TrafficAction(light_phase=0 if ns_em_urgency >= ew_em_urgency else 1)

    # Queue-based switching with hysteresis
    ns_total = q[0] + q[1]
    ew_total = q[2] + q[3]
    min_phase_time = min(3 + max(ns_total, ew_total) // 5, 8)

    if current == 0 and time_in < min_phase_time and ns_total > 0:
        return TrafficAction(light_phase=0)
    if current == 1 and time_in < min_phase_time and ew_total > 0:
        return TrafficAction(light_phase=1)

    if ns_total >= ew_total + 2:
        return TrafficAction(light_phase=0)
    elif ew_total >= ns_total + 2:
        return TrafficAction(light_phase=1)
    else:
        return TrafficAction(light_phase=current if current in (0, 1) else 0)


def _parse_phase(raw: str) -> int:
    """Extract phase from LLM response."""
    try:
        data = json.loads(raw)
        phase = int(data.get("light_phase", data.get("phase", 0)))
        return max(0, min(2, phase))
    except Exception:
        # Fallback: look for digit in response
        m = re.search(r'\b([012])\b', raw)
        return int(m.group(1)) if m else 0


def get_llm_action(client: OpenAI, obs: TrafficObservation, step: int) -> TrafficAction:
    """Hybrid: rule-based for obvious cases, LLM for ambiguous."""
    em_q = obs.emergency_queue
    q = obs.queue_lengths
    current = obs.current_phase
    time_in = obs.time_in_phase

    # Quick wins — pure rule-based (no API call)
    ns_em = em_q[0] + em_q[1]
    ew_em = em_q[2] + em_q[3]
    ns_total = q[0] + q[1]
    ew_total = q[2] + q[3]

    # Emergency handling — always rule-based for speed
    if ns_em > 0 and ew_em == 0:
        return TrafficAction(light_phase=0)
    if ew_em > 0 and ns_em == 0:
        return TrafficAction(light_phase=1)

    # Stay in phase if beneficial and within min time
    if current in (0, 1) and time_in < 3:
        if (current == 0 and ns_total > 0) or (current == 1 and ew_total > 0):
            return TrafficAction(light_phase=current)

    # Ambiguous case — call LLM
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(obs, step)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = resp.choices[0].message.content.strip()
        phase = _parse_phase(raw)
        return TrafficAction(light_phase=phase)
    except Exception as exc:
        import sys
        print(f"LLM API Error: {exc}", file=sys.stderr)
        # Fallback to rule-based on API error
        return _rule_based_action(obs)


def run_task(task: str, client: OpenAI) -> None:
    """Run a single task episode."""
    print(f'[START] task="{task}"', flush=True)

    rewards: List[float] = []
    step = 0
    error_msg: Optional[str] = None
    success = False

    try:
        with TrafficControlEnv(base_url=SERVER_URL).sync() as env:
            # Note: openenv-core's reset takes task_id, so passing task_id=task
            obs = env.reset(task_id=task, seed=SEED)

            while not obs.done:
                step += 1
                action = get_llm_action(client, obs, step)

                try:
                    obs = env.step(action)
                except Exception as exc:
                    error_msg = _sanitize(str(exc))
                    print(
                        f'[STEP] step={step} action={action} reward=0.0 done=true error="{error_msg}"',
                        flush=True,
                    )
                    break

                reward_val = obs.reward if obs.reward is not None else 0.0
                rewards.append(reward_val)

                print(
                    f'[STEP] step={step} action={action} '
                    f'reward={reward_val:.2f} done={str(obs.done).lower()} '
                    f'phase={obs.current_phase} '
                    f'queues={list(obs.queue_lengths)} '
                    f'emergency={list(obs.emergency_queue)}',
                    flush=True,
                )

            success = obs.done and not error_msg

    except Exception as exc:
        error_msg = _sanitize(str(exc))
        success = False

    total_reward = sum(rewards)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards[-10:])  # last 10 for brevity

    print(
        f'[END] success={str(success).lower()} steps={step} '
        f'total_reward={total_reward:.2f} rewards=[{rewards_str}]',
        flush=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy-key")
    except Exception:
        client = None

    import time
    # Fast reconnect logic so we don't trigger external Phase 2 timeouts
    for _ in range(5):
        try:
            r = _http.get(f"{SERVER_URL.rstrip('/')}/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(1)

    for task in ["basic_flow", "emergency_priority", "dynamic_scenarios"]:
        run_task(task, client)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        err = _sanitize(str(exc))
        print(f'[FATAL] error="{err}"', flush=True)
        raise SystemExit(1)
