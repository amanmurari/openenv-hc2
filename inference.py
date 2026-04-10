"""
Inference Script — Autonomous Traffic Control OpenEnv Environment
=================================================================
Advanced Hybrid Agent: combines optimized rule engine + LLM for ambiguous cases.

Mandatory env variables (injected by validator):
    API_BASE_URL   LLM proxy endpoint (MUST use validator's LiteLLM proxy)
    MODEL_NAME     Model identifier
    HF_TOKEN       Hugging Face API token / LiteLLM proxy key

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
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

SEED        = 42
MAX_TOKENS  = 64
TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# Enhanced LLM Prompt — scoring-aware
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an elite Autonomous Traffic Control AI managing a 4-way intersection.

    OBJECTIVE: Maximise your SCORE by balancing throughput, emergency response,
    efficiency, and stability (avoid unnecessary phase switching).

    PHASES:
      0 = North-South Green  (N/S vehicles may pass, up to 3 per direction per step)
      1 = East-West Green    (E/W vehicles may pass, up to 3 per direction per step)
      2 = All Red            (no vehicles pass — use ONLY for emergency clearance)

    SCORING COMPONENTS (what you're graded on):
      - Throughput: vehicles cleared per step (target ≥ 1.8/step)
      - Emergency response: clear emergency vehicles FAST (avg delay < 3 steps)
      - Efficiency: minimize total waiting time
      - Adaptability: DON'T switch phases too often (penalty for over-switching!)
      - Stability: staying in a productive phase is rewarded

    CRITICAL RULES (apply in strict order):
      1. EMERGENCY VEHICLES: If ANY emergency vehicle is waiting (emergency_queue > 0),
         switch to their direction IMMEDIATELY. Emergency delay is heavily penalized.
         Higher urgency = more critical. Urgency 8-10 is catastrophic.

      2. STAY IN PRODUCTIVE PHASE: If current phase is clearing vehicles AND
         queue has traffic, STAY. Each switch costs 2 yellow steps of zero throughput.

      3. MINIMUM PHASE TIME: Stay at least 3-5 steps in a phase (more for deeper queues).
         If time_in_phase < 3 and current direction has traffic, STAY.

      4. SWITCH ON IMBALANCE: Only switch when the OTHER direction has 3+ more
         vehicles than current direction. Small differences don't justify the switch cost.

      5. EMPTY QUEUE: If current direction queue = 0 and other direction > 0, switch.

      6. NEVER use phase 2 (All Red) unless ALL queues are empty.

    OUTPUT: Exactly one JSON object, no markdown, no explanation:
      {"light_phase": <0, 1, or 2>}
""").strip()


def _build_prompt(obs: TrafficObservation, step: int, total_rewards: float) -> str:
    """Build a rich prompt with scoring context for the LLM."""
    q    = obs.queue_lengths
    em_q = obs.emergency_queue
    em_u = obs.emergency_urgency

    # Queue trend info
    trend = getattr(obs, 'queue_trend', [0, 0, 0, 0])
    avg_wait = getattr(obs, 'avg_wait_time', 0.0)

    ns_total = q[0] + q[1]
    ew_total = q[2] + q[3]
    ns_em_total = em_q[0] + em_q[1]
    ew_em_total = em_q[2] + em_q[3]

    return textwrap.dedent(f"""
        STEP {step} | Cumulative reward: {total_rewards:.1f}

        CURRENT STATE:
          Active phase      : {obs.current_phase} (0=NS Green, 1=EW Green, 2=All Red)
          Steps in phase    : {obs.time_in_phase}

        QUEUES:
          Regular vehicles  : N={q[0]}, S={q[1]}, E={q[2]}, W={q[3]}
          → NS total: {ns_total}  |  EW total: {ew_total}  |  Difference: {abs(ns_total - ew_total)}
          Queue trend (Δ)   : N={trend[0]:+d}, S={trend[1]:+d}, E={trend[2]:+d}, W={trend[3]:+d}
          Avg wait time     : {avg_wait:.1f} steps

        EMERGENCIES:
          Emergency queue   : N={em_q[0]}, S={em_q[1]}, E={em_q[2]}, W={em_q[3]}
          Emergency urgency : N={em_u[0]}, S={em_u[1]}, E={em_u[2]}, W={em_u[3]}
          → NS emergencies: {ns_em_total}  |  EW emergencies: {ew_em_total}

        DECISION: {{"light_phase": <0, 1, or 2>}}
    """).strip()

# ---------------------------------------------------------------------------
# Advanced rule-based engine — score-maximizing
# ---------------------------------------------------------------------------

class SmartRuleEngine:
    """Stateful rule-based agent that tracks history for better decisions."""

    def __init__(self):
        self.phase_change_count = 0
        self.total_steps = 0
        self.last_3_queues: List[List[int]] = []

    def decide(self, obs: TrafficObservation) -> TrafficAction:
        self.total_steps += 1

        em_q = obs.emergency_queue
        em_u = obs.emergency_urgency
        q    = obs.queue_lengths
        current = obs.current_phase
        time_in = obs.time_in_phase

        # Track queue history for trend analysis
        total_q = [q[i] + em_q[i] for i in range(4)]
        self.last_3_queues.append(total_q)
        if len(self.last_3_queues) > 3:
            self.last_3_queues.pop(0)

        # ── Rule 1: EMERGENCY PRIORITY (highest priority, override everything) ──
        ns_em_score = em_u[0] + em_u[1] + em_q[0] * 3 + em_q[1] * 3
        ew_em_score = em_u[2] + em_u[3] + em_q[2] * 3 + em_q[3] * 3

        if ns_em_score > 0 or ew_em_score > 0:
            target = 0 if ns_em_score >= ew_em_score else 1
            if target != current:
                self.phase_change_count += 1
            return TrafficAction(light_phase=target)

        # ── Rule 2: EMPTY CURRENT DIRECTION → instant switch ──
        ns_total = q[0] + q[1]
        ew_total = q[2] + q[3]

        if current == 0 and ns_total == 0 and ew_total > 0:
            self.phase_change_count += 1
            return TrafficAction(light_phase=1)
        if current == 1 and ew_total == 0 and ns_total > 0:
            self.phase_change_count += 1
            return TrafficAction(light_phase=0)

        # ── Rule 3: DYNAMIC MINIMUM PHASE TIME ──
        # Deeper queues → stay longer to maximize throughput before switching
        current_dir_queue = ns_total if current == 0 else ew_total
        other_dir_queue = ew_total if current == 0 else ns_total

        # Adaptive min time: 3 base + 1 per 4 vehicles, capped at 10
        min_phase_time = min(3 + current_dir_queue // 4, 10)

        if time_in < min_phase_time and current_dir_queue > 0:
            return TrafficAction(light_phase=current if current in (0, 1) else 0)

        # ── Rule 4: ADAPTABILITY-AWARE SWITCHING THRESHOLD ──
        # The more we've already switched, the higher the threshold to switch again
        switch_rate = self.phase_change_count / max(self.total_steps, 1)
        # Base threshold is 3 vehicles; increases if we're switching too much
        switch_threshold = 3 + int(switch_rate * 10)

        if current == 0 and ew_total >= ns_total + switch_threshold:
            self.phase_change_count += 1
            return TrafficAction(light_phase=1)
        elif current == 1 and ns_total >= ew_total + switch_threshold:
            self.phase_change_count += 1
            return TrafficAction(light_phase=0)

        # ── Rule 5: QUEUE TREND ANALYSIS ──
        # If other direction's queue is growing fast (trend > 0 for last 3 steps)
        if len(self.last_3_queues) >= 3:
            if current == 0:
                ew_growing = all(
                    self.last_3_queues[i][2] + self.last_3_queues[i][3] <=
                    self.last_3_queues[i+1][2] + self.last_3_queues[i+1][3]
                    for i in range(len(self.last_3_queues) - 1)
                )
                if ew_growing and ew_total > ns_total and time_in >= 3:
                    self.phase_change_count += 1
                    return TrafficAction(light_phase=1)
            elif current == 1:
                ns_growing = all(
                    self.last_3_queues[i][0] + self.last_3_queues[i][1] <=
                    self.last_3_queues[i+1][0] + self.last_3_queues[i+1][1]
                    for i in range(len(self.last_3_queues) - 1)
                )
                if ns_growing and ns_total > ew_total and time_in >= 3:
                    self.phase_change_count += 1
                    return TrafficAction(light_phase=0)

        # ── Default: STAY in current phase for stability bonus ──
        return TrafficAction(light_phase=current if current in (0, 1) else 0)


# ---------------------------------------------------------------------------
# Sanitize error strings
# ---------------------------------------------------------------------------

def _sanitize(s: str) -> str:
    """Strip newlines, carriage returns, and problematic characters for output."""
    return s.replace('\n', ' ').replace('\r', ' ').replace('"', "'").replace('\\', '')

# ---------------------------------------------------------------------------
# LLM action with smart fallback
# ---------------------------------------------------------------------------

_rule_engine = SmartRuleEngine()


def get_llm_action(
    client: OpenAI,
    obs: TrafficObservation,
    step: int,
    total_rewards: float,
) -> TrafficAction:
    """
    Hybrid approach:
    - Use rules for clear-cut decisions (saves API calls + faster)
    - Use LLM for ambiguous situations (close queues, complex emergencies)
    """
    q    = obs.queue_lengths
    em_q = obs.emergency_queue
    current = obs.current_phase

    ns_total = q[0] + q[1]
    ew_total = q[2] + q[3]
    ns_em = sum(em_q[0:2])
    ew_em = sum(em_q[2:4])
    diff = abs(ns_total - ew_total)

    # ── FAST PATH: clear-cut decisions → use rules (no LLM call needed) ──

    # Emergency vehicles → always rules (speed critical, don't wait for LLM)
    if ns_em > 0 or ew_em > 0:
        return _rule_engine.decide(obs)

    # Empty current direction → obvious switch
    if current == 0 and ns_total == 0 and ew_total > 0:
        return _rule_engine.decide(obs)
    if current == 1 and ew_total == 0 and ns_total > 0:
        return _rule_engine.decide(obs)

    # Large imbalance → obvious switch
    if diff >= 5:
        return _rule_engine.decide(obs)

    # Very early in phase → obviously stay
    if obs.time_in_phase < 3:
        return _rule_engine.decide(obs)

    # ── SLOW PATH: ambiguous situation → ask LLM ──
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_prompt(obs, step, total_rewards)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            timeout=30,
        )
        data_str = (resp.choices[0].message.content or "").strip()
        match = re.search(r'\{[^}]*\}', data_str.replace('\n', ' '))
        data  = json.loads(match.group(0) if match else data_str)
        phase = max(0, min(2, int(data.get("light_phase", obs.current_phase))))

        # Update rule engine state even when using LLM
        _rule_engine.total_steps += 1
        if phase != current:
            _rule_engine.phase_change_count += 1

        return TrafficAction(light_phase=phase)
    except Exception:
        # LLM failed — use optimized rule-based agent as fallback
        return _rule_engine.decide(obs)

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
    global _rule_engine
    _rule_engine = SmartRuleEngine()  # Fresh engine per task

    print(f"[START] task={task_id} env=traffic_control model={MODEL_NAME}", flush=True)

    rewards: List[float] = []
    success = False
    step = 0
    total_rewards = 0.0

    try:
        with TrafficControlEnv(base_url=SERVER_URL).sync() as env:
            step_result = env.reset(task_id=task_id, seed=SEED)
            step = 0
            broke_on_error = False

            while not step_result.done:
                obs = step_result.observation
                step += 1

                action = get_llm_action(client, obs, step, total_rewards)
                action_str = f"light_phase({action.light_phase})"

                try:
                    step_result = env.step(action)
                    reward_val = step_result.reward if step_result.reward is not None else 0.0
                    rewards.append(reward_val)
                    total_rewards += reward_val
                    done_val = str(step_result.done).lower()

                    error_val = "null"
                    if hasattr(step_result, 'info') and step_result.info:
                        err = step_result.info.get('error')
                        if err:
                            error_val = _sanitize(str(err))

                    print(
                        f"[STEP] step={step} action={action_str} "
                        f"reward={reward_val:.2f} done={done_val} error={error_val}",
                        flush=True,
                    )
                except Exception as exc:
                    env_err = _sanitize(str(exc))
                    rewards.append(0.0)
                    print(
                        f"[STEP] step={step} action={action_str} "
                        f"reward=0.00 done=true error={env_err}",
                        flush=True,
                    )
                    broke_on_error = True
                    break

            success = not broke_on_error

    except Exception as exc:
        err_msg = _sanitize(str(exc))
        if step == 0:
            step = 1
            rewards.append(0.0)
            print(f"[STEP] step=1 action=null reward=0.00 done=true error={err_msg}", flush=True)
        success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"

    print(
        f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task in ["basic_flow", "emergency_priority", "dynamic_scenarios"]:
        run_task(task, client)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        err = _sanitize(str(exc))
        print(f"[START] task=unknown env=traffic_control model={MODEL_NAME}", flush=True)
        print(f"[STEP] step=1 action=null reward=0.00 done=true error={err}", flush=True)
        print(f"[END] success=false steps=1 rewards=0.00", flush=True)
