"""
FastAPI app for the Autonomous Traffic Control OpenEnv environment.

Endpoints provided automatically by openenv-core create_app():
  POST /reset    – start a new episode
  POST /step     – execute one action
  GET  /state    – episode-level cumulative state
  GET  /schema   – action / observation JSON schemas
  WS   /ws       – WebSocket for persistent sessions
  GET  /health   – liveness probe
  GET  /docs     – Swagger UI

Custom endpoints added here:
  POST /grade    – run the automated task grader (returns 0-1 score)
  GET  /ui       – Gradio testing interface

Usage:
    # From traffic_control/ directory:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
    python -m traffic_control.server.app
"""

from __future__ import annotations

import sys
import os

# Ensure this package's parent is on sys.path so relative package imports work
# regardless of from where uvicorn is invoked.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # traffic_control/
_ROOT    = os.path.dirname(_PKG_DIR)                                      # openv/
for _p in (_PKG_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from typing import Optional
from openenv.core.env_server.http_server import create_app
from fastapi import Request
from fastapi.responses import HTMLResponse

# All imports from within traffic_control/ only
from traffic_control.models import TrafficAction, TrafficObservation
from traffic_control.environment import TrafficControlEnvironment
from traffic_control.tasks import grade as run_grader
from traffic_control.dashboard import render_intersection, observation_to_render_state
from traffic_control.analytics import get_history
from traffic_control.arena import get_arena


# ---------------------------------------------------------------------------
# 1. Standard OpenEnv app
# ---------------------------------------------------------------------------

app = create_app(
    TrafficControlEnvironment,
    TrafficAction,
    TrafficObservation,
    env_name="traffic_control",
    max_concurrent_envs=4,
)


# ---------------------------------------------------------------------------
# 2. /grade endpoint & root for HF Spaces
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    """Hugging Face Spaces healthcheck requires a 200 OK on the root path."""
    return {"status": "ok", "message": "Autonomous Traffic Control OpenEnv"}

@app.post("/grade", tags=["eval"])
async def grade(request: Request):
    """
    Run the automated grader for the environment's current state.

    Body (all optional):
        task_id, total_vehicles_passed, total_emergency_passed,
        total_waiting_time, total_collisions, total_emergency_delay,
        total_phase_changes, step_count
    """
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    task_id               = body.get("task_id", "basic_flow")
    total_vehicles        = int(body.get("total_vehicles_passed", 0))
    total_emergency       = int(body.get("total_emergency_passed", 0))
    total_waiting         = float(body.get("total_waiting_time", 0.0))
    total_collisions      = int(body.get("total_collisions", 0))
    total_emergency_delay = float(body.get("total_emergency_delay", 0.0))
    total_phase_changes   = int(body.get("total_phase_changes", 0))
    step_count            = int(body.get("step_count", 1))

    result = run_grader(
        task_id,
        total_vehicles_passed=total_vehicles,
        total_emergency_passed=total_emergency,
        total_waiting_time=total_waiting,
        total_collisions=total_collisions,
        total_emergency_delay=total_emergency_delay,
        total_phase_changes=total_phase_changes,
        step_count=step_count,
    )

    return {
        "task_id":  task_id,
        "score":    result.score,
        "metrics":  result.metrics,
        "feedback": result.feedback,
    }


# ---------------------------------------------------------------------------
# Dashboard endpoints
# ---------------------------------------------------------------------------

@app.get("/dashboard", tags=["dashboard"])
async def dashboard():
    """Serve the live dashboard HTML page."""
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Traffic Control Dashboard</title>
    <style>
        body {
            font-family: system-ui, -apple-system, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 10px;
        }
        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        button, select {
            padding: 10px 20px;
            font-size: 14px;
            border-radius: 6px;
            border: none;
            cursor: pointer;
        }
        button {
            background: #3b82f6;
            color: white;
        }
        button:hover {
            background: #2563eb;
        }
        select {
            background: #1e293b;
            color: #e2e8f0;
            border: 1px solid #475569;
        }
        .dashboard-container {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
        }
        .visualization {
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
        }
        .visualization svg {
            width: 100%;
            height: auto;
            border-radius: 8px;
        }
        .stats-panel {
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
        }
        .stat-item {
            margin-bottom: 15px;
            padding: 12px;
            background: #0f172a;
            border-radius: 8px;
        }
        .stat-label {
            font-size: 12px;
            color: #94a3b8;
            text-transform: uppercase;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #22c55e;
        }
        .stat-value.warning {
            color: #eab308;
        }
        .stat-value.danger {
            color: #ef4444;
        }
        .phase-indicator {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .phase-ns { background: #22c55e; color: #064e3b; }
        .phase-ew { background: #3b82f6; color: #1e3a8a; }
        .phase-red { background: #ef4444; color: #7f1d1d; }
        .queue-bar {
            height: 8px;
            background: #334155;
            border-radius: 4px;
            margin-top: 5px;
            overflow: hidden;
        }
        .queue-fill {
            height: 100%;
            background: linear-gradient(90deg, #22c55e, #eab308, #ef4444);
            transition: width 0.3s;
        }
        .auto-toggle {
            display: flex;
            align-items: center;
            gap: 10px;
            justify-content: center;
            margin-top: 10px;
        }
        .toggle-switch {
            position: relative;
            width: 50px;
            height: 24px;
            background: #475569;
            border-radius: 12px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .toggle-switch.active {
            background: #22c55e;
        }
        .toggle-slider {
            position: absolute;
            top: 2px;
            left: 2px;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            transition: transform 0.3s;
        }
        .toggle-switch.active .toggle-slider {
            transform: translateX(26px);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚦 Autonomous Traffic Control Dashboard</h1>
        
        <div class="controls">
            <select id="taskSelect">
                <option value="basic_flow">Basic Flow</option>
                <option value="emergency_priority">Emergency Priority</option>
                <option value="dynamic_scenarios">Dynamic Scenarios</option>
            </select>
            <button onclick="resetEnv()">🔄 Reset</button>
            <button onclick="stepOnce()">▶️ Step Once</button>
            <div class="auto-toggle">
                <span>Auto Run</span>
                <div class="toggle-switch" id="autoToggle" onclick="toggleAuto()">
                    <div class="toggle-slider"></div>
                </div>
            </div>
            <button onclick="setPhase(0)">🟢 NS Green</button>
            <button onclick="setPhase(1)">🟢 EW Green</button>
            <button onclick="setPhase(2)">🔴 All Red</button>
        </div>
        
        <div class="dashboard-container">
            <div class="visualization">
                <div id="svgContainer">Loading...</div>
            </div>
            
            <div class="stats-panel">
                <h3>📊 Live Statistics</h3>
                
                <div class="stat-item">
                    <div class="stat-label">Current Phase</div>
                    <div id="phaseValue" class="stat-value">-</div>
                </div>
                
                <div class="stat-item">
                    <div class="stat-label">Step Count</div>
                    <div id="stepValue" class="stat-value">0</div>
                </div>
                
                <div class="stat-item">
                    <div class="stat-label">Total Reward</div>
                    <div id="rewardValue" class="stat-value">0.00</div>
                </div>
                
                <div class="stat-item">
                    <div class="stat-label">Vehicles Passed</div>
                    <div id="vehiclesValue" class="stat-value">0</div>
                </div>
                
                <div class="stat-item">
                    <div class="stat-label">Emergency Vehicles</div>
                    <div id="emergencyValue" class="stat-value">0</div>
                </div>
                
                <div class="stat-item">
                    <div class="stat-label">Queue Depths</div>
                    <div id="queueValue" style="font-size: 14px;">N:0 S:0 E:0 W:0</div>
                    <div class="queue-bar">
                        <div class="queue-fill" id="queueBar" style="width: 0%"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let autoRunning = false;
        let autoInterval = null;
        let totalReward = 0;
        let totalVehicles = 0;
        let totalEmergency = 0;
        let stepCount = 0;
        
        async function resetEnv() {
            const task = document.getElementById('taskSelect').value;
            try {
                await fetch('/reset', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({task_id: task, seed: 42})
                });
                totalReward = 0;
                totalVehicles = 0;
                totalEmergency = 0;
                stepCount = 0;
                updateDashboard();
            } catch (e) {
                console.error('Reset failed:', e);
            }
        }
        
        async function stepOnce() {
            try {
                const response = await fetch('/step', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action: {light_phase: 0}})
                });
                const data = await response.json();
                updateStats(data);
                updateDashboard();
            } catch (e) {
                console.error('Step failed:', e);
            }
        }
        
        async function setPhase(phase) {
            try {
                const response = await fetch('/step', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action: {light_phase: phase}})
                });
                const data = await response.json();
                updateStats(data);
                updateDashboard();
            } catch (e) {
                console.error('Set phase failed:', e);
            }
        }
        
        function toggleAuto() {
            autoRunning = !autoRunning;
            document.getElementById('autoToggle').classList.toggle('active', autoRunning);
            
            if (autoRunning) {
                autoInterval = setInterval(stepOnce, 1000);
            } else {
                clearInterval(autoInterval);
            }
        }
        
        function updateStats(data) {
            if (data.reward) totalReward += data.reward;
            if (data.metadata && data.metadata.vehicles_passed) {
                totalVehicles = data.metadata.total_vehicles_passed || totalVehicles;
                totalEmergency = data.metadata.total_emergency_passed || totalEmergency;
            }
            stepCount++;
            
            document.getElementById('stepValue').textContent = stepCount;
            document.getElementById('rewardValue').textContent = totalReward.toFixed(2);
            document.getElementById('vehiclesValue').textContent = totalVehicles;
            document.getElementById('emergencyValue').textContent = totalEmergency;
            
            if (data.observation) {
                const obs = data.observation;
                const phaseNames = {0: 'NS GREEN', 1: 'EW GREEN', 2: 'ALL RED', 3: 'NS YELLOW', 4: 'EW YELLOW'};
                document.getElementById('phaseValue').textContent = phaseNames[obs.current_phase] || 'UNKNOWN';
                
                const queues = obs.queue_lengths;
                const totalQueue = queues.reduce((a, b) => a + b, 0) + obs.emergency_queue.reduce((a, b) => a + b, 0);
                document.getElementById('queueValue').textContent = `N:${queues[0]} S:${queues[1]} E:${queues[2]} W:${queues[3]}`;
                document.getElementById('queueBar').style.width = Math.min(totalQueue * 5, 100) + '%';
            }
        }
        
        async function updateDashboard() {
            try {
                const response = await fetch('/state');
                const state = await response.json();
                
                const svgResponse = await fetch('/dashboard/svg', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(state)
                });
                const svgData = await svgResponse.json();
                document.getElementById('svgContainer').innerHTML = svgData.svg;
            } catch (e) {
                console.error('Dashboard update failed:', e);
            }
        }
        
        // Initial load
        resetEnv();
    </script>
</body>
</html>'''
    return HTMLResponse(content=html_content)


@app.post("/dashboard/svg", tags=["dashboard"])
async def dashboard_svg(request: Request):
    """Generate SVG visualization from current state."""
    try:
        state_data = await request.json()
        render_state = observation_to_render_state(
            state_data.get("observation", {}),
            total_vehicles=state_data.get("total_vehicles_passed", 0),
            total_emergency=state_data.get("total_emergency_passed", 0),
            step=state_data.get("step_count", 0),
            reward=state_data.get("reward", 0.0),
        )
        svg = render_intersection(render_state)
        return {"svg": svg}
    except Exception as e:
        return {"error": str(e), "svg": ""}


# ---------------------------------------------------------------------------
# Analytics endpoints
# ---------------------------------------------------------------------------

@app.get("/analytics/summary", tags=["analytics"])
async def analytics_summary(task_id: Optional[str] = None):
    """Get summary statistics of all recorded episodes."""
    return get_history().get_summary(task_id)


@app.get("/analytics/episodes", tags=["analytics"])
async def list_episodes():
    """List all recorded episodes."""
    history = get_history()
    return {
        "episodes": [
            {
                "episode_id": e.episode_id,
                "task_id": e.task_id,
                "steps": e.steps,
                "total_reward": round(e.total_reward, 2),
                "avg_reward_per_step": round(e.avg_reward_per_step, 4),
            }
            for e in history.episodes
        ]
    }


@app.get("/analytics/episodes/{episode_id}", tags=["analytics"])
async def get_episode(episode_id: str):
    """Get detailed metrics for a specific episode."""
    details = get_history().get_episode_details(episode_id)
    if details:
        return details
    return {"error": "Episode not found"}


# ---------------------------------------------------------------------------
# Arena endpoints
# ---------------------------------------------------------------------------

@app.post("/arena/run", tags=["arena"])
async def arena_run(request: Request):
    """Run agent comparison in the arena."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    
    task_id = body.get("task_id", "basic_flow")
    agents = body.get("agents", None)  # List of agent types or None for all
    runs_per_agent = int(body.get("runs_per_agent", 1))
    
    arena = get_arena()
    result = await arena.run_comparison(
        task_id=task_id,
        agents=agents,
        runs_per_agent=runs_per_agent,
    )
    return result


@app.get("/arena/agents", tags=["arena"])
async def list_agents():
    """List available agents in the arena."""
    return {
        "agents": [
            {"id": "rule_based", "name": "Smart Rule-Based", "description": "Optimized rule-based controller"},
            {"id": "random", "name": "Random Baseline", "description": "Random action selector"},
            {"id": "round_robin", "name": "Round Robin", "description": "Simple alternating controller"},
        ]
    }


@app.get("/arena/results", tags=["arena"])
async def arena_results():
    """Get all historical arena results."""
    arena = get_arena()
    return {
        "total_comparisons": len(arena.results),
        "recent_results": [
            {
                "agent": r.agent_type,
                "episode": r.episode_id,
                "score": r.score,
                "reward": round(r.total_reward, 2),
                "steps": r.steps,
            }
            for r in arena.results[-20:]
        ],
    }


# ---------------------------------------------------------------------------
# 3. Gradio UI (mounted at /ui)
# ---------------------------------------------------------------------------

try:
    import gradio as gr
    import requests as _req

    # Use SERVER_PORT env var (default 8000) so the UI works on any port
    _PORT = int(os.environ.get("PORT", os.environ.get("SERVER_PORT", "8000")))
    _SELF_BASE = f"http://127.0.0.1:{_PORT}"

    def _reset_env(task_id: str):
        try:
            r = _req.post(
                f"{_SELF_BASE}/reset",
                json={"task_id": task_id, "seed": 42},
                timeout=10,
            )
            return r.json() if r.status_code == 200 else {"error": r.text}
        except Exception as exc:
            return {"error": str(exc)}

    def _step_env(phase: str):
        try:
            # openenv-core wraps the action under an "action" key
            r = _req.post(
                f"{_SELF_BASE}/step",
                json={"action": {"light_phase": int(phase)}},
                timeout=10,
            )
            return r.json() if r.status_code == 200 else {"error": r.text}
        except Exception as exc:
            return {"error": str(exc)}

    def _get_state():
        try:
            r = _req.get(f"{_SELF_BASE}/state", timeout=10)
            return r.json() if r.status_code == 200 else {"error": r.text}
        except Exception as exc:
            return {"error": str(exc)}

    with gr.Blocks(title="Traffic Control UI") as _ui:
        gr.Markdown("# 🚦 Autonomous Traffic Control — Testing UI")
        gr.Markdown("Interact with the OpenEnv HTTP API live.")

        with gr.Row():
            _task = gr.Dropdown(
                choices=["basic_flow", "emergency_priority", "dynamic_scenarios"],
                value="basic_flow",
                label="Task ID",
            )
            _reset_btn = gr.Button("🔄 Reset")
            _state_btn = gr.Button("📊 State")

        with gr.Row():
            _phase = gr.Radio(
                choices=[("0 – NS Green", "0"), ("1 – EW Green", "1"), ("2 – All Red", "2")],
                value="0",
                label="Next Action (Light Phase)",
            )
            _step_btn = gr.Button("▶️ Step", variant="primary")

        _out = gr.JSON(label="API Response")

        _reset_btn.click(_reset_env, inputs=[_task],  outputs=[_out])
        _step_btn.click(_step_env,  inputs=[_phase],  outputs=[_out])
        _state_btn.click(_get_state,                  outputs=[_out])

    app = gr.mount_gradio_app(app, _ui, path="/ui")

except ImportError:
    # Gradio is optional; server still works without it
    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: start uvicorn server. Reads --host and --port from CLI args."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Traffic Control OpenEnv Server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--workers", type=int, default=1)
    args, _ = parser.parse_known_args()  # ignore unknown args from uv

    uvicorn.run(
        "traffic_control.server.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
