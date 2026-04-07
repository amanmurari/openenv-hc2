"""
Typed data models for the Autonomous Traffic Control Environment.

All Pydantic models extend openenv-core base types so the environment
is fully compliant with the OpenEnv specification.
"""

from typing import List, Optional, Any, Dict
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


# ---------------------------------------------------------------------------
# Phase / direction constants
# ---------------------------------------------------------------------------

PHASE_NS_GREEN  = 0   # North-South green, East-West red
PHASE_EW_GREEN  = 1   # East-West green, North-South red
PHASE_ALL_RED   = 2   # All approaches red (emergency clearance)
PHASE_NS_YELLOW = 3   # North-South transitioning to red
PHASE_EW_YELLOW = 4   # East-West transitioning to red

DIRECTION_NORTH = 0
DIRECTION_SOUTH = 1
DIRECTION_EAST  = 2
DIRECTION_WEST  = 3


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class TrafficAction(Action):
    """
    Agent action: set the desired traffic light phase.

    light_phase:
        0 = NS_GREEN  – North + South get green, East + West get red.
        1 = EW_GREEN  – East + West get green, North + South get red.
        2 = ALL_RED   – All approaches red; use for emergency clearance.
    """
    light_phase: int = Field(
        ...,
        ge=0, le=2,
        description="Desired light phase: 0=NS_GREEN | 1=EW_GREEN | 2=ALL_RED",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TrafficObservation(Observation):
    """
    Full observation returned by reset() and step().

    Directions index: 0=North, 1=South, 2=East, 3=West
    The `done`, `reward`, and `metadata` fields are inherited from
    openenv.core.env_server.types.Observation.
    """

    # -- Traffic-light state --
    current_phase: int = Field(
        default=0,
        description="Active light phase: 0=NS_GREEN|1=EW_GREEN|2=ALL_RED|3=NS_YELLOW|4=EW_YELLOW",
    )
    time_in_phase: int = Field(
        default=0,
        description="Steps elapsed since last phase change",
    )

    # -- Vehicle queues (one per direction [N, S, E, W]) --
    queue_lengths: List[int] = Field(
        default_factory=lambda: [0, 0, 0, 0],
        description="Regular vehicle count per approach [N, S, E, W]",
    )
    emergency_queue: List[int] = Field(
        default_factory=lambda: [0, 0, 0, 0],
        description="Emergency vehicle count per approach [N, S, E, W]",
    )
    emergency_urgency: List[int] = Field(
        default_factory=lambda: [0, 0, 0, 0],
        description="Max urgency (0-10) of waiting emergency vehicles per approach",
    )

    # -- Flow metrics for this step --
    vehicles_passed: int = Field(default=0, description="Regular vehicles cleared this step")
    emergency_passed: int = Field(default=0, description="Emergency vehicles cleared this step")

    # -- Penalty signals --
    total_waiting_time: float = Field(
        default=0.0,
        description="Sum of per-vehicle waiting increments this step",
    )
    collision: bool = Field(default=False, description="Gridlock-induced collision flag")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TrafficState(State):
    """
    Episode-level cumulative state, returned by state().

    The `episode_id` and `step_count` fields are inherited from
    openenv.core.env_server.types.State.
    """
    task_id: str = Field(default="basic_flow", description="Active task ID")

    # Cumulative episode metrics
    total_vehicles_passed: int = Field(default=0)
    total_emergency_passed: int = Field(default=0)
    total_waiting_time: float = Field(default=0.0)
    total_emergency_delay: float = Field(
        default=0.0,
        description="Steps emergency vehicles spent waiting",
    )
    total_collisions: int = Field(default=0)
    total_phase_changes: int = Field(default=0)
