"""HTTP/WebSocket client for the Autonomous Traffic Control environment."""

from typing import Any, Dict
from openenv.core.env_client import EnvClient, StepResult

from .models import TrafficAction, TrafficObservation, TrafficState


class TrafficControlEnv(EnvClient[TrafficAction, TrafficObservation, TrafficState]):
    """
    Client for the Autonomous Traffic Control environment.

    Inherits all openenv-core EnvClient functionality:
    - async context manager
    - .sync() wrapper for synchronous use
    - reset() / step() / state()
    - from_docker_image() for local Docker deployment
    - from_env() for HuggingFace Space deployment

    Example (sync):
        with TrafficControlEnv(base_url="http://localhost:8000").sync() as env:
            obs = env.reset(task_id="basic_flow", seed=42)
            while not obs.done:
                obs = env.step(TrafficAction(light_phase=0))

    Example (async):
        async with TrafficControlEnv(base_url="http://localhost:8000") as env:
            obs = await env.reset(task_id="emergency_priority", seed=0)
            while not obs.done:
                obs = await env.step(TrafficAction(light_phase=1))
    """

    def _step_payload(self, action: TrafficAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TrafficObservation]:
        obs = TrafficObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TrafficState:
        return TrafficState(**payload)
