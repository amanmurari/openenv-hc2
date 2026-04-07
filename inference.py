"""
Inference Script — Autonomous Traffic Control OpenEnv Environment
=================================================================
Mandatory env variables:
    API_BASE_URL   LLM endpoint  (default: https://api.openai.com/v1)
    MODEL_NAME     Model to use  (default: gpt-4.1-mini)
    HF_TOKEN       Your Hugging Face / LLM API key  ← REQUIRED

Optional:
    SERVER_URL     Running env server (default: http://localhost:8000)

Run:
    HF_TOKEN=<key> python inference.py
    HF_TOKEN=<key> SERVER_URL=http://localhost:8000 python inference.py
"""

import os
import sys
import json
import textwrap

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

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN:     str = os.getenv("HF_TOKEN", "")
SERVER_URL:   str = os.getenv("SERVER_URL", "http://localhost:8000")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

SEED        = 42
MAX_TOKENS  = 32
TEMPERATURE = 0.0

llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        data  = json.loads(resp.choices[0].message.content or "{}")
        phase = int(data.get("light_phase", obs.current_phase))
        phase = max(0, min(2, phase))
        return TrafficAction(light_phase=phase)
    except Exception:
        return _rule_based_action(obs)

# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> None:
    print(f"[START] task={task_id} env=traffic-control model={MODEL_NAME}")

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
                        f"reward={reward_val:.2f} done={done_str} error={error_msg}"
                    )
                except Exception as exc:
                    env_error = str(exc).replace('"', "'").replace("\\", "")
                    print(
                        f"[STEP] step={step} action={action_str} "
                        f"reward=0.00 done=true error={env_error}"
                    )
                    break

                step += 1

            success = True

    except Exception as exc:
        print(f"[STEP] step=0 action=none reward=0.00 done=true error={exc}")
        success = False

    success_str  = "true" if success else "false"
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={success_str} steps={len(rewards)} rewards={rewards_str}")


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
