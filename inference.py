"""Inference Script — Autonomous Traffic Control OpenEnv Environment"""

import os
import sys
import json
import time
import urllib.request
from typing import List, Optional

# Allow imports from repo root so both container-root and package contexts work
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openai import OpenAI

try:
    from traffic_control.client import TrafficControlEnv
    from traffic_control.models import TrafficAction, TrafficObservation
except ImportError:
    from client import TrafficControlEnv  # type: ignore
    from models import TrafficAction, TrafficObservation  # type: ignore

# ---------------------------------------------------------------------------
# Environment variables — MUST be injected by the hackathon validator
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
SERVER_URL   = os.getenv("SERVER_URL", "http://localhost:7860")
SEED         = 42
MAX_TOKENS   = 64
TEMPERATURE  = 0.0

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


def get_llm_action(client: OpenAI, obs: TrafficObservation, step: int) -> TrafficAction:
    """Call LLM proxy for a traffic phase decision."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_prompt(obs, step)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    raw = resp.choices[0].message.content.strip()
    phase = _parse_phase(raw)
    return TrafficAction(light_phase=phase)


def _wait_for_server(url: str, timeout: int = 60) -> None:
    """Block until the env server is healthy or timeout expires."""
    health_url = url.rstrip("/") + "/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    # If the server never came up, log and continue anyway
    print(f"[WARN] Server not healthy after {timeout}s — proceeding anyway", flush=True)


def run_task(task: str, client: OpenAI) -> dict:
    """Run a single task episode."""
    print(f'[START] task={task} env=traffic_control model={MODEL_NAME}', flush=True)

    rewards: List[float] = []
    step = 0
    last_error: Optional[str] = None
    done = False

    try:
        with TrafficControlEnv(base_url=SERVER_URL).sync() as env:
            # env.reset() returns a StepResult; unwrap the observation
            reset_result = env.reset(task_id=task, seed=SEED)
            obs: TrafficObservation = reset_result.observation
            done = reset_result.done

            while not done:
                step += 1

                # Call the LLM proxy — this is the call the validator monitors
                action = get_llm_action(client, obs, step)
                action_str = f"light_phase={action.light_phase}"

                try:
                    # env.step() returns a StepResult; unwrap the observation
                    result = env.step(action)
                    obs        = result.observation   # TrafficObservation
                    reward_val = result.reward if result.reward is not None else 0.0
                    rewards.append(reward_val)
                    done       = result.done
                    last_error = None
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
    score        = min(1.0, max(0.0, total_reward / max_possible)) if max_possible > 0 else 0.0

    print(
        f'[END] success={str(success).lower()} steps={step} '
        f'score={score:.2f} rewards={rewards_str}',
        flush=True,
    )

    return {"success": success, "steps": step, "rewards": rewards}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Main entry point."""
    # Wait for the env server to be ready before starting inference
    _wait_for_server(SERVER_URL)

    # Initialize OpenAI client with hackathon-injected proxy credentials
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    tasks = ["basic_flow", "emergency_priority", "dynamic_scenarios"]
    for task in tasks:
        run_task(task, client)


if __name__ == "__main__":
    main()
