---
title: Autonomous Traffic Control Environment
emoji: 🚦
colorFrom: red
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - traffic-control
  - emergency-vehicles
  - autonomous-systems
---

# 🚦 Autonomous Traffic Control — OpenEnv Environment

An LLM-driven reinforcement learning environment for autonomous traffic signal control at a 4-way intersection. Built for the **Meta × PyTorch × SST × OpenEnv Hackathon**.

- **HF Space:** [amanmurari/sst-hack](https://huggingface.co/spaces/amanmurari/sst-hack)
- **GitHub:** [amanmurari/openenv-hc2](https://github.com/amanmurari/openenv-hc2)

---

## Overview

An LLM agent controls traffic signals to maximise vehicle throughput while prioritising emergency vehicles. The environment features:

- Sinusoidal traffic wave patterns (realistic rush-hour simulation)
- Emergency vehicles with escalating urgency (urgency^1.5 penalty per waiting step)
- Yellow-light transition state machine
- Traffic surge events in hard tasks
- Multi-objective grading aligned with real traffic KPIs

---

## Tasks

| Task | Difficulty | Steps | Key Challenge |
|---|---|---|---|
| `basic_flow` | Easy | 200 | Maximise throughput (target 1.8 veh/step) |
| `emergency_priority` | Medium | 300 | Clear emergencies fast (avg delay < 3 steps) |
| `dynamic_scenarios` | Hard | 400 | Surge traffic + simultaneous emergencies, no collisions |

---

## Action & Observation Space

### Action
```python
TrafficAction(light_phase: int)
# 0 = NS_GREEN  (North-South green, East-West red)
# 1 = EW_GREEN  (East-West green, North-South red)
# 2 = ALL_RED   (All red — emergency clearance)
```

### Observation
```python
TrafficObservation(
    current_phase: int,          # Active phase (0-4, incl. yellow transitions)
    time_in_phase: int,          # Steps held in current phase
    queue_lengths: List[int],    # Regular vehicle queue [N, S, E, W]
    emergency_queue: List[int],  # Emergency vehicle count [N, S, E, W]
    emergency_urgency: List[int],# Max urgency 0-10 per approach
    vehicles_passed: int,        # Regular vehicles cleared this step
    emergency_passed: int,       # Emergency vehicles cleared this step
    avg_wait_time: float,        # Avg waiting time across all queued vehicles
    queue_trend: List[int],      # Queue growth since last step [N, S, E, W]
    collision: bool,             # Gridlock-induced collision flag
    done: bool,
    reward: float,
)
```

---

## Reward Function

```
+0.30  × regular vehicles cleared per step
+12.0  × emergency vehicles cleared per step
-0.08  × total vehicles waiting (queue pressure)
-(urgency^1.5) × 0.5  per waiting emergency vehicle (every step!)
-0.50 to -2.0  for unnecessary phase switch (proportional to empty-queue ratio)
+0.05  stability bonus when traffic flows without switching
-200   gridlock collision (episode-ending catastrophe)
```

---

## Grading Weights

### basic_flow
```
score = throughput_score × 0.60 + efficiency_score × 0.40 + stability_bonus
throughput_score = min(vehicles_per_step / 1.8, 1.0)
efficiency_score = 1 / (1 + avg_waiting × 0.1)
stability_bonus  = max(0, 0.05 × (1 − min(switch_rate × 4, 1.0)))
```

### emergency_priority
```
score = throughput × 0.30 + em_rate × 0.35 + delay × 0.20 + efficiency × 0.15
em_rate = min(emergency_cleared_per_step / (1/20), 1.0)
delay   = max(0, 1 − avg_em_delay / 12)
```

### dynamic_scenarios
```
score = throughput × 0.25 + em_rate × 0.30 + delay × 0.20
      + efficiency × 0.15 + adaptability × 0.10
adaptability = 1 / (1 + phase_changes_per_step × 0.5)
```

---

## Agent Architecture

The inference agent uses a **hybrid heuristic + LLM** architecture:

1. **Heuristic recommender** — computes directional pressure using the actual reward formula (`urgency^1.5 × 0.5`), applies 5 priority rules (critical emergency, moderate emergency, hysteresis, pressure-based switch, default hold).

2. **Live score projection** — computes current projected grading scores (throughput, emergency rate, delay, efficiency, adaptability) from `env.state()` and includes them in every LLM prompt.

3. **Chain-of-thought LLM** — the model reasons through scoring implications then outputs `{"light_phase": N}` on the final line.

4. **Heuristic fallback** — if LLM output is unparseable, silently falls back to the heuristic. No crashes, no missed steps.

---

## Quick Start

### Connect to the live HF Space

```python
from traffic_control.client import TrafficControlEnv
from traffic_control.models import TrafficAction

with TrafficControlEnv(base_url="https://amanmurari-sst-hack.hf.space").sync() as env:
    result = env.reset(task_id="basic_flow", seed=42)
    obs = result.observation

    while not result.done:
        action = TrafficAction(light_phase=0)   # replace with your agent
        result = env.step(action)
        obs = result.observation
        print(f"Cleared: {obs.vehicles_passed} regular, {obs.emergency_passed} emergency | reward={result.reward:.2f}")
```

### Run inference locally

```bash
# Set required env vars
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="hf_..."
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export SERVER_URL="http://localhost:7860"

python inference.py
```

### Build and run with Docker

```bash
docker build -t traffic-control-env .
docker run -p 7860:7860 traffic-control-env
```

---

## API Endpoints

Once the server is running at `http://localhost:7860`:

| Endpoint | Description |
|---|---|
| `GET /health` | Health check — returns `{"status": "ok"}` |
| `POST /reset` | Reset episode — body: `{"task_id": "basic_flow", "seed": 42}` |
| `POST /step` | Execute action — body: `{"light_phase": 0}` |
| `GET /state` | Cumulative episode state |
| `WS /ws` | WebSocket endpoint for low-latency multi-step sessions |
| `GET /web` | Interactive web dashboard |
| `GET /docs` | OpenAPI / Swagger docs |

---

## Project Structure

```
traffic_control/
├── inference.py          # LLM agent (heuristic + chain-of-thought LLM)
├── client.py             # TrafficControlEnv WebSocket client
├── models.py             # TrafficAction / TrafficObservation / TrafficState
├── environment.py        # Core simulation engine
├── tasks.py              # Task graders (basic_flow, emergency_priority, dynamic_scenarios)
├── dashboard.py          # Web UI dashboard
├── analytics.py          # Episode analytics
├── arena.py              # Multi-agent arena
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml        # Package metadata
├── Dockerfile            # Container (port 7860)
└── server/
    ├── app.py            # FastAPI app (HTTP + WebSocket)
    └── traffic_control_environment.py
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | LLM proxy endpoint (injected by validator) |
| `API_KEY` | Yes | Proxy API key (injected by validator) |
| `HF_TOKEN` | Alt | Hugging Face token (used if `API_KEY` not set) |
| `MODEL_NAME` | No | LLM model (default: `Qwen/Qwen2.5-72B-Instruct`) |
| `SERVER_URL` | No | Env server URL (default: `http://localhost:7860`) |

---

## Stdout Format

```
[START] task=basic_flow env=traffic_control model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=light_phase=0 reward=0.65 done=false error=null
[STEP] step=2 action=light_phase=0 reward=0.80 done=false error=null
...
[END] success=true steps=200 score=0.847 rewards=0.65,0.80,...
```
