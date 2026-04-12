"""
Inference Script — Autonomous Traffic Control OpenEnv Environment
Strictly follows validator spec from hackathon guidelines.
"""

import os
import sys
import json
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
# Configuration - EXACTLY per spec: defaults for API_BASE_URL and MODEL_NAME
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ["API_KEY"]

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")
SEED = 42
MAX_TOKENS = 64
TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# LLM Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Autonomous Traffic Signal Controller.

PHASES:
  0 = North-South Green
  1 = East-West Green  
  2 = All Red

SCORING:
  +0.2 per regular vehicle cleared
  +10.0 per emergency vehicle cleared
  -0.4 * urgency per step emergency waits
  -0.5 for switching to empty queue

OUTPUT: {"light_phase": 0, 1, or 2} only JSON, no other text."""


def _build_prompt(obs: TrafficObservation, step: int) -> str:
    return (
        f"Step {step}, phase={obs.current_phase}, time={obs.time_in_phase}\n"
        f"Queues N,S,E,W: {list(obs.queue_lengths)}\n"
        f"Emergency N,S,E,W: {list(obs.emergency_queue)} urgency={list(obs.emergency_urgency)}\n"
        f"What phase? Return only JSON: {{\"light_phase\": 0, 1, or 2}}"
    )


def _sanitize(s: str) -> str:
    """Remove characters that break log parsing."""
    return s.replace('"', "'").replace("\n", " ")




def _parse_phase(raw: str) -> int:
    """Extract phase from LLM response."""
    try:
        data = json.loads(raw)
        phase = int(data.get("light_phase", data.get("phase", 0)))
        return max(0, min(2, phase))
    except Exception:
        import re
        m = re.search(r'\b([012])\b', raw)
        return int(m.group(1)) if m else 0


def get_llm_action(client: OpenAI, obs: TrafficObservation, step: int, model: str) -> TrafficAction:
    """Call LLM for decision."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(obs, step)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    raw = resp.choices[0].message.content.strip()
    phase = _parse_phase(raw)
    return TrafficAction(light_phase=phase)


def run_task(task: str, client: OpenAI, model: str) -> dict:
    """Run a single task episode."""
    print(f'[START] task={task} env=traffic_control model={model}', flush=True)

    rewards: List[float] = []
    step = 0
    last_error: Optional[str] = None
    done = False

    try:
        with TrafficControlEnv(base_url=SERVER_URL).sync() as env:
            obs = env.reset(task_id=task, seed=SEED)

            while not obs.done:
                step += 1
                action = get_llm_action(client, obs, step, model)
                action_str = f"light_phase={action.light_phase}"

                try:
                    obs = env.step(action)
                    reward_val = obs.reward if obs.reward is not None else 0.0
                    rewards.append(reward_val)
                    done = obs.done
                    last_error = None
                except Exception as exc:
                    reward_val = 0.0
                    done = True
                    last_error = _sanitize(str(exc))

                error_str = "null" if last_error is None else f'"{last_error}"'
                print(
                    f'[STEP] step={step} action={action_str} reward={reward_val:.2f} done={str(done).lower()} error={error_str}',
                    flush=True,
                )

                if done:
                    break

    except Exception as exc:
        last_error = _sanitize(str(exc))

    success = done and last_error is None
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    
    # Calculate normalized score [0, 1]
    total_reward = sum(rewards)
    max_possible = step * 10.0  # Approximate max per step
    score = min(1.0, max(0.0, total_reward / max_possible)) if max_possible > 0 else 0.0
    
    print(
        f'[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={rewards_str}',
        flush=True,
    )
    
    return {"success": success, "steps": step, "rewards": rewards}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Main entry point - reads env vars directly as validator requires."""
    # CRITICAL: Read environment variables directly here for validator detection
    api_base = os.environ["API_BASE_URL"]
    api_key = os.environ["API_KEY"]
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    print(f"[INIT] API_BASE_URL={api_base[:30]}...", flush=True)
    print(f"[INIT] API_KEY present={bool(api_key)}", flush=True)
    print(f"[INIT] MODEL_NAME={model}", flush=True)
    
    # Initialize OpenAI client with directly-read env vars
    client = OpenAI(base_url=api_base, api_key=api_key)
    print(f"[INIT] Client ready", flush=True)

    tasks = ["basic_flow", "emergency_priority", "dynamic_scenarios"]
    for task in tasks:
        run_task(task, client, model)


if __name__ == "__main__":
    main()
