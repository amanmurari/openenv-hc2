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

from openenv.core.env_server.http_server import create_app
from fastapi import Request

# All imports from within traffic_control/ only
from traffic_control.models import TrafficAction, TrafficObservation
from traffic_control.environment import TrafficControlEnvironment
from traffic_control.tasks import grade as run_grader


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
# 2. /grade endpoint
# ---------------------------------------------------------------------------

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

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start uvicorn server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
