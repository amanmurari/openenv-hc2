"""
Core simulation for the Autonomous Traffic Control Environment.

Implements openenv-core's Environment interface so it works directly
with create_app() — no adapters needed.

Simulates a 4-way intersection with:
  - Time-varying traffic wave patterns (sinusoidal arrival rates)
  - Directional traffic imbalance (rush-hour asymmetry)
  - Emergency vehicles with urgency levels (0-10) that escalate over time
  - Yellow-light transition state machine (2-step yellow)
  - Traffic-surge events (hard task only)
  - Multi-objective reward function aligned with grading weights
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
# Task configurations — enhanced with wave/imbalance parameters
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
        # Wave pattern: sinusoidal variation in arrival rate
        "wave_amplitude":          0.15,   # ±15% variation
        "wave_period":             40,     # steps per wave cycle
        # Directional imbalance: multiplier for NS vs EW
        "ns_bias":                 1.2,    # NS gets 20% more traffic
        "ew_bias":                 0.8,
    },
    "emergency_priority": {
        "vehicle_arrival_rate":    0.5,
        "emergency_arrival_rate":  0.015,
        "emergency_urgency_range": (7, 10),
        "max_steps":               300,
        "max_queue_per_lane":      20,
        "surge_probability":       0.0,
        "surge_multiplier":        1.0,
        "wave_amplitude":          0.20,
        "wave_period":             50,
        "ns_bias":                 1.0,
        "ew_bias":                 1.0,
    },
    "dynamic_scenarios": {
        "vehicle_arrival_rate":    0.7,
        "emergency_arrival_rate":  0.035,
        "emergency_urgency_range": (8, 10),
        "max_steps":               400,
        "max_queue_per_lane":      30,
        "surge_probability":       0.04,
        "surge_multiplier":        3.0,
        "wave_amplitude":          0.25,
        "wave_period":             60,
        "ns_bias":                 1.3,
        "ew_bias":                 0.7,
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
        self._prev_queue_lengths: List[int] = [0, 0, 0, 0]
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
        self._prev_queue_lengths = [0, 0, 0, 0]
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

        # Snapshot queue lengths before this step for trend tracking
        self._prev_queue_lengths = [
            len(q) for q in self._queues
        ]

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

    def _get_wave_rate(self, base_rate: float) -> float:
        """Apply sinusoidal wave pattern to arrival rate."""
        amp    = self._cfg.get("wave_amplitude", 0.0)
        period = self._cfg.get("wave_period", 40)
        if amp <= 0 or period <= 0:
            return base_rate
        wave = math.sin(2 * math.pi * self._step_count / period)
        return max(0.05, base_rate * (1.0 + amp * wave))

    def _spawn_vehicles(self) -> None:
        base_arr = self._cfg["vehicle_arrival_rate"]
        em       = self._cfg["emergency_arrival_rate"]
        urg      = self._cfg["emergency_urgency_range"]
        surge_p  = self._cfg["surge_probability"]
        surge_m  = self._cfg["surge_multiplier"]
        max_q    = self._cfg["max_queue_per_lane"]
        ns_bias  = self._cfg.get("ns_bias", 1.0)
        ew_bias  = self._cfg.get("ew_bias", 1.0)

        # Apply wave pattern
        arr = self._get_wave_rate(base_arr)

        surge_dir   = -1
        surge_extra = 0
        if surge_p > 0.0 and self._rng.random() < surge_p:
            surge_dir   = self._rng.randint(0, 3)
            surge_extra = max(0, int(self._rng.gauss(3, 1) * surge_m))

        for d in range(4):
            # Directional bias: NS directions (0,1) vs EW (2,3)
            dir_bias = ns_bias if d in (0, 1) else ew_bias
            n = self._poisson(arr * dir_bias)
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
                    # Urgency escalates over time — waiting makes it worse
                    v.urgency = min(10, v.urgency + (1 if v.waiting_time % 5 == 0 else 0))
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
        # --- Throughput reward (aligned with grading target ~1.8-2.0 veh/step) ---
        r  = vehicles_passed  * 0.30
        r += emergency_passed * 12.0

        # --- Waiting penalty (progressive) ---
        r -= waiting_delta * 0.08

        # --- Emergency urgency penalty (super-linear: urgency^1.5) ---
        for d in range(4):
            for v in self._queues[d]:
                if v.vehicle_type == VehicleType.EMERGENCY:
                    r -= (v.urgency ** 1.5) * 0.5

        # --- Collision is catastrophic ---
        if collision:
            r -= 200.0

        # --- Phase change penalty (proportional to wasted switch) ---
        if phase_changed:
            p = int(self._current_phase)
            new_dir_queue = 0
            if p == PHASE_NS_GREEN:
                new_dir_queue = len(self._queues[0]) + len(self._queues[1])
            elif p == PHASE_EW_GREEN:
                new_dir_queue = len(self._queues[2]) + len(self._queues[3])

            total_queue = sum(len(q) for q in self._queues)
            if total_queue > 0:
                empty_ratio = 1.0 - (new_dir_queue / total_queue)
                r -= 0.5 + empty_ratio * 1.5  # heavier penalty for switching to emptier side
            else:
                r -= 0.5

        # --- Stability bonus: reward NOT switching when traffic is flowing ---
        if not phase_changed and vehicles_passed > 0:
            r += 0.05

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

        # Compute queue trend (current - previous)
        current_totals = [len(q) for q in self._queues]
        queue_trend = [
            current_totals[i] - self._prev_queue_lengths[i]
            for i in range(4)
        ]

        # Compute average wait time
        all_waits = [v.waiting_time for q in self._queues for v in q]
        avg_wait = sum(all_waits) / max(len(all_waits), 1) if all_waits else 0.0

        return TrafficObservation(
            current_phase=int(self._current_phase),
            time_in_phase=self._time_in_phase,
            queue_lengths=queue_lengths,
            emergency_queue=emergency_queue,
            emergency_urgency=emergency_urgency,
            vehicles_passed=vehicles_passed,
            emergency_passed=emergency_passed,
            avg_wait_time=round(avg_wait, 2),
            queue_trend=queue_trend,
            total_vehicles_passed_cumulative=self._total_vehicles_passed,
            total_emergency_passed_cumulative=self._total_emergency_passed,
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
