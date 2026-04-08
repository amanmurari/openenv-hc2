"""
Inference Script — Autonomous Traffic Control OpenEnv Environment
=================================================================
Mandatory env variables:
    API_BASE_URL   LLM endpoint  (default: https://api.openai.com/v1)
    MODEL_NAME     Model to use  (default: gpt-4.1-mini)
    API_KEY        Your LLM API key / LiteLLM Proxy Key ← REQUIRED (can fallback to HF_TOKEN)

Optional:
    SERVER_URL     Running env server (default: http://localhost:8000)

Run:
    API_KEY=<key> python inference.py
    API_KEY=<key> SERVER_URL=http://localhost:8000 python inference.py
"""

import os
import sys
import json
import textwrap
import requests as _http

# Allow running directly from traffic_control/ OR from its parent
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

# Import from within the self-contained package
try:
    from traffic_control.client import TrafficControlEnv
    from traffic_control.models import TrafficAction, TrafficObservation
except ImportError:
    from client import TrafficControlEnv
    from models import TrafficAction, TrafficObservation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")

# The testing environment proxies use "API_KEY", but the pre-validation wants "HF_TOKEN"
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY  = os.getenv("API_KEY", HF_TOKEN)

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

if not API_KEY:
    raise ValueError("API_KEY or HF_TOKEN environment variable is required")

SEED        = 42
MAX_TOKENS  = 32
TEMPERATURE = 0.0

llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an Autonomous Traffic Control AI managing a 4-way intersection.

    OBJECTIVE: Maximise vehicle throughput and prioritise emergency vehicles.

    PHASES:
      0 = North-South Green  (N/S vehicles may pass)
      1 = East-West Green    (E/W vehicles may pass)
      2 = All Red            (no vehicles pass — use only for emergency clearance)

    STRATEGY:
      1. If any emergency vehicles are waiting, switch to their direction immediately.
      2. Otherwise, switch to the direction with the most queued vehicles.
      3. Avoid changing phase too frequently (wait ≥ 4 steps per phase).

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

        Output exactly: {{"light_phase": 0}}
    """).strip()

# ---------------------------------------------------------------------------
# Rule-based fallback (used when LLM call fails)
# ---------------------------------------------------------------------------

def _rule_based_action(obs: TrafficObservation) -> TrafficAction:
    """Simple heuristic: emergency first, else highest queue."""
    em_q = obs.emergency_queue
    q    = obs.queue_lengths

    # Emergency vehicle present?
    if sum(em_q) > 0:
        if em_q[0] + em_q[1] >= em_q[2] + em_q[3]:
            return TrafficAction(light_phase=0)  # NS Green
        else:
            return TrafficAction(light_phase=1)  # EW Green

    # Highest queue direction
    ns_total = q[0] + q[1]
    ew_total = q[2] + q[3]
    if obs.current_phase == 0 and obs.time_in_phase < 4:
        return TrafficAction(light_phase=0)
    if obs.current_phase == 1 and obs.time_in_phase < 4:
        return TrafficAction(light_phase=1)
    return TrafficAction(light_phase=0 if ns_total >= ew_total else 1)

# ---------------------------------------------------------------------------
# LLM action
# ---------------------------------------------------------------------------

def get_llm_action(obs: TrafficObservation) -> TrafficAction:
    try:
        resp = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_prompt(obs)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        data_str = resp.choices[0].message.content or "{}"
        
        # Try to parse the JSON manually since we removed response_format
        import re
        match = re.search(r'\{.*\}', data_str.replace('\n', ' '))
        if match:
            data = json.loads(match.group(0))
        else:
            data = json.loads(data_str)
            
        phase = int(data.get("light_phase", obs.current_phase))
        phase = max(0, min(2, phase))
        return TrafficAction(light_phase=phase)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        # If it fails, fallback to rule-based BUT we also raise so the orchestrator 
        # knows it failed instead of silently passing with 0 proxy usage.
        raise exc

# ---------------------------------------------------------------------------
# Grade fetcher  (calls /grade after episode ends)
# ---------------------------------------------------------------------------

def _fetch_score(task_id: str, state_payload: dict) -> float:
    """Call the /grade endpoint and return a clamped (0.001, 0.999) score."""
    try:
        r = _http.post(
            f"{SERVER_URL}/grade",
            json={
                "task_id":               task_id,
                "total_vehicles_passed": state_payload.get("total_vehicles_passed", 0),
                "total_emergency_passed":state_payload.get("total_emergency_passed", 0),
                "total_waiting_time":    state_payload.get("total_waiting_time", 0.0),
                "total_collisions":      state_payload.get("total_collisions", 0),
                "total_emergency_delay": state_payload.get("total_emergency_delay", 0.0),
                "total_phase_changes":   state_payload.get("total_phase_changes", 0),
                "step_count":            max(state_payload.get("step_count", 1), 1),
            },
            timeout=10,
        )
        if r.status_code == 200:
            raw = float(r.json().get("score", 0.5))
            return max(0.001, min(0.999, raw))
    except Exception:
        pass
    return 0.5  # safe fallback


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> None:
    print(f"[START] task={task_id} env=traffic-control model={MODEL_NAME}", flush=True)

    rewards: list[float] = []
    success = False

    try:
        with TrafficControlEnv(base_url=SERVER_URL).sync() as env:
            step_result = env.reset(task_id=task_id, seed=SEED)
            step = 1

            while not step_result.done:
                obs       = step_result.observation
                error_msg = "null"

                try:
                    action = get_llm_action(obs)
                except Exception as exc:
                    error_msg = str(exc).replace('"', "'").replace("\\", "")
                    action    = _rule_based_action(obs)

                action_str = f"TrafficAction(light_phase={action.light_phase})"

                try:
                    step_result = env.step(action)
                    done_str    = "true" if step_result.done else "false"
                    reward_val  = step_result.reward if step_result.reward is not None else 0.0
                    rewards.append(reward_val)
                    print(
                        f"[STEP] step={step} action={action_str} "
                        f"reward={reward_val:.2f} done={done_str} error={error_msg}",
                        flush=True
                    )
                except Exception as exc:
                    env_error = str(exc).replace('"', "'").replace("\\", "")
                    print(
                        f"[STEP] step={step} action={action_str} "
                        f"reward=0.00 done=true error={env_error}",
                        flush=True
                    )
                    break

                step += 1

            success = True

    except Exception as exc:
        print(f"[STEP] step=0 action=none reward=0.00 done=true error={exc}", flush=True)
        success = False

    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"

    # Fetch final grade score from /grade endpoint
    score = 0.5
    try:
        state_resp = _http.get(f"{SERVER_URL}/state", timeout=10)
        if state_resp.status_code == 200:
            score = _fetch_score(task_id, state_resp.json())
    except Exception:
        pass

    print(f"[END] success={success_str} steps={len(rewards)} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    tasks = ["basic_flow", "emergency_priority", "dynamic_scenarios"]
    for task in tasks:
        run_task(task)
        print()  # blank line between tasks


if __name__ == "__main__":
    main()
