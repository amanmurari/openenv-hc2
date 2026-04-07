"""
Core simulation for the Autonomous Traffic Control Environment.

Implements openenv-core's Environment interface so it works directly
with create_app() — no adapters needed.

Simulates a 4-way intersection with:
  - Poisson vehicle arrivals per approach
  - Emergency vehicles with urgency levels (0-10)
  - Yellow-light transition state machine (2-step yellow)
  - Traffic-surge events (hard task only)
  - Multi-objective reward function
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

from openenv.core.env_server.interfaces import Environment

from .models import (
    TrafficAction,
    TrafficObservation,
    TrafficState,
    PHASE_NS_GREEN,
    PHASE_EW_GREEN,
    PHASE_ALL_RED,
    PHASE_NS_YELLOW,
    PHASE_EW_YELLOW,
)


# ---------------------------------------------------------------------------
# Internal enums
# ---------------------------------------------------------------------------

class VehicleType(IntEnum):
    CAR       = 0
    BUS       = 1
    EMERGENCY = 2


class Direction(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST  = 2
    WEST  = 3


class LightPhase(IntEnum):
    NS_GREEN  = PHASE_NS_GREEN
    EW_GREEN  = PHASE_EW_GREEN
    ALL_RED   = PHASE_ALL_RED
    NS_YELLOW = PHASE_NS_YELLOW
    EW_YELLOW = PHASE_EW_YELLOW


# ---------------------------------------------------------------------------
# Phase transition tables
# ---------------------------------------------------------------------------

PHASE_ALLOWS: Dict[LightPhase, Set[int]] = {
    LightPhase.NS_GREEN:  {Direction.NORTH, Direction.SOUTH},
    LightPhase.EW_GREEN:  {Direction.EAST,  Direction.WEST},
    LightPhase.ALL_RED:   set(),
    LightPhase.NS_YELLOW: {Direction.NORTH, Direction.SOUTH},
    LightPhase.EW_YELLOW: {Direction.EAST,  Direction.WEST},
}

PHASE_FLOW_RATE: Dict[LightPhase, int] = {
    LightPhase.NS_GREEN:  3,
    LightPhase.EW_GREEN:  3,
    LightPhase.ALL_RED:   0,
    LightPhase.NS_YELLOW: 1,
    LightPhase.EW_YELLOW: 1,
}

YELLOW_DURATION = 2


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, dict] = {
    "basic_flow": {
        "vehicle_arrival_rate":    0.4,
        "emergency_arrival_rate":  0.0,
        "emergency_urgency_range": (0, 0),
        "max_steps":               200,
        "max_queue_per_lane":      20,
        "surge_probability":       0.0,
        "surge_multiplier":        1.0,
    },
    "emergency_priority": {
        "vehicle_arrival_rate":    0.5,
        "emergency_arrival_rate":  0.015,
        "emergency_urgency_range": (7, 10),
        "max_steps":               300,
        "max_queue_per_lane":      20,
        "surge_probability":       0.0,
        "surge_multiplier":        1.0,
    },
    "dynamic_scenarios": {
        "vehicle_arrival_rate":    0.7,
        "emergency_arrival_rate":  0.035,
        "emergency_urgency_range": (8, 10),
        "max_steps":               400,
        "max_queue_per_lane":      30,
        "surge_probability":       0.04,
        "surge_multiplier":        3.0,
    },
}


# ---------------------------------------------------------------------------
# Internal vehicle dataclass
# ---------------------------------------------------------------------------

@dataclass
class Vehicle:
    vehicle_type: VehicleType
    direction: Direction
    waiting_time: int = 0
    urgency: int = 0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class TrafficControlEnvironment(Environment):
    """
    OpenEnv-compliant Autonomous Traffic Control environment.

    Inherits from openenv.core.env_server.interfaces.Environment,
    making it compatible with openenv-core's create_app() factory.

    Methods
    -------
    reset(seed, episode_id, **kwargs) -> TrafficObservation
    step(action)                      -> TrafficObservation
    state (property)                  -> TrafficState
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = "basic_flow") -> None:
        if task_id not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {list(TASK_CONFIGS.keys())}"
            )
        self.task_id = task_id
        self._cfg = TASK_CONFIGS[task_id]
        self._rng = random.Random()

        self._episode_id: str = ""
        self._step_count: int = 0
        self._queues: List[List[Vehicle]] = [[] for _ in range(4)]
        self._current_phase: LightPhase = LightPhase.NS_GREEN
        self._time_in_phase: int = 0
        self._pending_phase: Optional[int] = None

        self._total_vehicles_passed: int   = 0
        self._total_emergency_passed: int  = 0
        self._total_waiting_time: float    = 0.0
        self._total_emergency_delay: float = 0.0
        self._total_collisions: int        = 0
        self._total_phase_changes: int     = 0

    # ------------------------------------------------------------------
    # openenv-core Environment interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> TrafficObservation:
        """Start a fresh episode."""
        if task_id and task_id in TASK_CONFIGS:
            self.task_id = task_id
            self._cfg = TASK_CONFIGS[task_id]

        self._rng         = random.Random(seed)
        self._episode_id  = episode_id or str(uuid.uuid4())
        self._step_count  = 0
        self._queues      = [[] for _ in range(4)]
        self._current_phase = LightPhase.NS_GREEN
        self._time_in_phase = 0
        self._pending_phase = None

        self._total_vehicles_passed   = 0
        self._total_emergency_passed  = 0
        self._total_waiting_time      = 0.0
        self._total_emergency_delay   = 0.0
        self._total_collisions        = 0
        self._total_phase_changes     = 0

        return self._build_obs(0, 0, 0.0, False, 0.0, False)

    def step(self, action: TrafficAction) -> TrafficObservation:  # type: ignore[override]
        """Execute one simulation step."""
        self._step_count += 1

        self._spawn_vehicles()
        phase_changed = self._apply_action(action)
        self._advance_phase()
        vehicles_passed, emergency_passed = self._flow_traffic()
        waiting_delta = self._tick_waiting_times()
        collision = self._check_collision()
        reward = self._compute_reward(
            vehicles_passed, emergency_passed, waiting_delta, collision, phase_changed
        )

        self._total_vehicles_passed  += vehicles_passed
        self._total_emergency_passed += emergency_passed
        self._total_waiting_time     += waiting_delta
        if collision:
            self._total_collisions += 1

        done = collision or self._step_count >= self._cfg["max_steps"]
        return self._build_obs(vehicles_passed, emergency_passed, waiting_delta, collision, reward, done)

    @property
    def state(self) -> TrafficState:
        """Return cumulative episode-level state."""
        return TrafficState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self.task_id,
            total_vehicles_passed=self._total_vehicles_passed,
            total_emergency_passed=self._total_emergency_passed,
            total_waiting_time=self._total_waiting_time,
            total_emergency_delay=self._total_emergency_delay,
            total_collisions=self._total_collisions,
            total_phase_changes=self._total_phase_changes,
        )

    # ------------------------------------------------------------------
    # Simulation internals
    # ------------------------------------------------------------------

    def _spawn_vehicles(self) -> None:
        arr     = self._cfg["vehicle_arrival_rate"]
        em      = self._cfg["emergency_arrival_rate"]
        urg     = self._cfg["emergency_urgency_range"]
        surge_p = self._cfg["surge_probability"]
        surge_m = self._cfg["surge_multiplier"]
        max_q   = self._cfg["max_queue_per_lane"]

        surge_dir   = -1
        surge_extra = 0
        if surge_p > 0.0 and self._rng.random() < surge_p:
            surge_dir   = self._rng.randint(0, 3)
            surge_extra = max(0, int(self._rng.gauss(3, 1) * surge_m))

        for d in range(4):
            n = self._poisson(arr)
            if d == surge_dir:
                n += surge_extra
            for _ in range(n):
                if len(self._queues[d]) < max_q:
                    vt = VehicleType.BUS if self._rng.random() < 0.10 else VehicleType.CAR
                    self._queues[d].append(Vehicle(vt, Direction(d)))

            if em > 0.0 and self._rng.random() < em:
                if len(self._queues[d]) < max_q:
                    urgency = self._rng.randint(urg[0], urg[1])
                    self._queues[d].insert(
                        0,
                        Vehicle(VehicleType.EMERGENCY, Direction(d), urgency=urgency),
                    )

    def _apply_action(self, action: TrafficAction) -> bool:
        req = action.light_phase
        if req not in (PHASE_NS_GREEN, PHASE_EW_GREEN, PHASE_ALL_RED):
            return False
        if self._current_phase in (LightPhase.NS_YELLOW, LightPhase.EW_YELLOW):
            return False
        current_base = int(self._current_phase)
        if current_base == req:
            return False

        self._total_phase_changes += 1
        self._pending_phase = req

        if req == PHASE_ALL_RED:
            self._current_phase = LightPhase.ALL_RED
            self._time_in_phase = 0
            self._pending_phase = None
        elif self._current_phase == LightPhase.NS_GREEN:
            self._current_phase = LightPhase.NS_YELLOW
            self._time_in_phase = 0
        elif self._current_phase == LightPhase.EW_GREEN:
            self._current_phase = LightPhase.EW_YELLOW
            self._time_in_phase = 0
        elif self._current_phase == LightPhase.ALL_RED:
            self._current_phase = LightPhase(req)
            self._time_in_phase = 0
            self._pending_phase = None

        return True

    def _advance_phase(self) -> None:
        self._time_in_phase += 1
        if self._current_phase in (LightPhase.NS_YELLOW, LightPhase.EW_YELLOW):
            if self._time_in_phase >= YELLOW_DURATION:
                target = self._pending_phase if self._pending_phase is not None else PHASE_ALL_RED
                self._current_phase = LightPhase(target)
                self._time_in_phase = 0
                self._pending_phase = None

    def _flow_traffic(self) -> Tuple[int, int]:
        allowed   = PHASE_ALLOWS[self._current_phase]
        flow_rate = PHASE_FLOW_RATE[self._current_phase]
        vehicles_passed  = 0
        emergency_passed = 0

        for d in allowed:
            queue      = self._queues[int(d)]
            passed_dir = 0
            while queue and passed_dir < flow_rate:
                vehicle = queue.pop(0)
                passed_dir += 1
                if vehicle.vehicle_type == VehicleType.EMERGENCY:
                    emergency_passed += 1
                else:
                    vehicles_passed += 1

        return vehicles_passed, emergency_passed

    def _tick_waiting_times(self) -> float:
        total = 0.0
        for d in range(4):
            for v in self._queues[d]:
                v.waiting_time += 1
                total += 1.0
                if v.vehicle_type == VehicleType.EMERGENCY:
                    self._total_emergency_delay += 1.0
        return total

    def _check_collision(self) -> bool:
        total_queued = sum(len(q) for q in self._queues)
        if total_queued > 40 and self._time_in_phase > 20:
            return self._rng.random() < 0.04
        return False

    def _compute_reward(
        self,
        vehicles_passed: int,
        emergency_passed: int,
        waiting_delta: float,
        collision: bool,
        phase_changed: bool,
    ) -> float:
        r  = vehicles_passed  * 0.20
        r += emergency_passed * 10.0
        r -= waiting_delta    * 0.05

        for d in range(4):
            for v in self._queues[d]:
                if v.vehicle_type == VehicleType.EMERGENCY:
                    r -= v.urgency * 0.4

        if collision:
            r -= 200.0

        if phase_changed:
            p = int(self._current_phase)
            if p == PHASE_NS_GREEN:
                if (len(self._queues[0]) + len(self._queues[1])) == 0:
                    r -= 0.5
            elif p == PHASE_EW_GREEN:
                if (len(self._queues[2]) + len(self._queues[3])) == 0:
                    r -= 0.5

        return r

    def _build_obs(
        self,
        vehicles_passed: int,
        emergency_passed: int,
        waiting_delta: float,
        collision: bool,
        reward: float,
        done: bool,
    ) -> TrafficObservation:
        queue_lengths     = []
        emergency_queue   = []
        emergency_urgency = []

        for d in range(4):
            reg   = sum(1 for v in self._queues[d] if v.vehicle_type != VehicleType.EMERGENCY)
            em    = sum(1 for v in self._queues[d] if v.vehicle_type == VehicleType.EMERGENCY)
            max_u = max(
                (v.urgency for v in self._queues[d] if v.vehicle_type == VehicleType.EMERGENCY),
                default=0,
            )
            queue_lengths.append(reg)
            emergency_queue.append(em)
            emergency_urgency.append(max_u)

        return TrafficObservation(
            current_phase=int(self._current_phase),
            time_in_phase=self._time_in_phase,
            queue_lengths=queue_lengths,
            emergency_queue=emergency_queue,
            emergency_urgency=emergency_urgency,
            vehicles_passed=vehicles_passed,
            emergency_passed=emergency_passed,
            total_waiting_time=waiting_delta,
            collision=collision,
            reward=reward,
            done=done,
            metadata={
                "step_count": self._step_count,
                "task_id":    self.task_id,
            },
        )

    def _poisson(self, lam: float) -> int:
        if lam <= 0.0:
            return 0
        threshold = math.exp(-lam)
        k, p = 0, 1.0
        while p > threshold:
            k += 1
            p *= self._rng.random()
        return k - 1
