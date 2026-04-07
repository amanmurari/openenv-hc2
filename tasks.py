"""
Task graders for the Autonomous Traffic Control OpenEnv environment.

Defines three tasks of increasing difficulty:
  1. basic_flow          – baseline throughput optimisation (Easy)
  2. emergency_priority  – emergency vehicle management + throughput (Medium)
  3. dynamic_scenarios   – surge-traffic + emergencies under hard constraints (Hard)

Each grader returns a GradeResult(score, metrics, feedback) with 0–1 score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class GradeResult:
    """Standardised grading result."""
    score: float                        # 0.0 – 1.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    feedback: str = ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def grade(
    task_id: str,
    *,
    total_vehicles_passed: int   = 0,
    total_emergency_passed: int  = 0,
    total_waiting_time: float    = 0.0,
    total_collisions: int        = 0,
    total_emergency_delay: float = 0.0,
    total_phase_changes: int     = 0,
    step_count: int              = 1,
) -> GradeResult:
    """Route to the appropriate task grader."""
    graders = {
        "basic_flow":         _grade_basic_flow,
        "emergency_priority": _grade_emergency_priority,
        "dynamic_scenarios":  _grade_dynamic_scenarios,
    }
    if task_id not in graders:
        return GradeResult(
            score=0.0,
            feedback=f"Unknown task_id '{task_id}'. Valid: {list(graders.keys())}",
        )
    return graders[task_id](
        total_vehicles_passed=total_vehicles_passed,
        total_emergency_passed=total_emergency_passed,
        total_waiting_time=total_waiting_time,
        total_collisions=total_collisions,
        total_emergency_delay=total_emergency_delay,
        total_phase_changes=total_phase_changes,
        step_count=max(step_count, 1),
    )


# ---------------------------------------------------------------------------
# Task 1 – Basic Flow  (Easy)
# ---------------------------------------------------------------------------

_BASIC_FLOW_TARGET_THROUGHPUT_PER_STEP = 1.8  # vehicles/step considered "perfect"


def _grade_basic_flow(
    *,
    total_vehicles_passed: int,
    total_waiting_time: float,
    total_collisions: int,
    step_count: int,
    **_ignored,
) -> GradeResult:
    throughput_per_step = total_vehicles_passed / step_count
    throughput_score    = min(throughput_per_step / _BASIC_FLOW_TARGET_THROUGHPUT_PER_STEP, 1.0)
    efficiency_score    = 1.0 / (1.0 + total_waiting_time / max(step_count, 1) * 0.1)
    collision_penalty   = 0.8 if total_collisions > 0 else 0.0

    raw   = throughput_score * 0.6 + efficiency_score * 0.4
    score = max(0.0, raw - collision_penalty)

    return GradeResult(
        score=round(score, 4),
        metrics={
            "throughput_per_step": round(throughput_per_step, 3),
            "throughput_score":    round(throughput_score,    4),
            "efficiency_score":    round(efficiency_score,    4),
            "total_collisions":    total_collisions,
            "collision_penalty":   collision_penalty,
        },
        feedback=(
            f"Throughput {throughput_per_step:.2f} veh/step "
            f"(target {_BASIC_FLOW_TARGET_THROUGHPUT_PER_STEP}). "
            + ("⚠ Collision penalty applied!" if total_collisions else "No collisions ✓.")
        ),
    )


# ---------------------------------------------------------------------------
# Task 2 – Emergency Priority  (Medium)
# ---------------------------------------------------------------------------

_EMERG_TARGET_DELAY_PER_VEHICLE = 3.0  # steps/emergency vehicle


def _grade_emergency_priority(
    *,
    total_vehicles_passed: int,
    total_emergency_passed: int,
    total_waiting_time: float,
    total_collisions: int,
    total_emergency_delay: float,
    step_count: int,
    **_ignored,
) -> GradeResult:
    throughput_per_step = total_vehicles_passed / step_count
    throughput_score    = min(throughput_per_step / 1.5, 1.0)

    # Emergency throughput score: 1.0 if ≥ 1 emergency vehicle cleared per 20 steps
    em_rate       = total_emergency_passed / step_count
    em_rate_score = min(em_rate / (1.0 / 20.0), 1.0)

    # Emergency delay score
    if total_emergency_passed > 0:
        avg_delay   = total_emergency_delay / total_emergency_passed
        delay_score = max(0.0, 1.0 - avg_delay / (_EMERG_TARGET_DELAY_PER_VEHICLE * 4))
    else:
        delay_score = 0.5

    efficiency_score  = 1.0 / (1.0 + total_waiting_time / max(step_count, 1) * 0.05)
    collision_penalty = 0.85 if total_collisions > 0 else 0.0

    raw   = (throughput_score * 0.30 + em_rate_score * 0.35 +
             delay_score      * 0.20 + efficiency_score * 0.15)
    score = max(0.0, raw - collision_penalty)

    avg_delay_str = (
        f"{total_emergency_delay / total_emergency_passed:.1f} steps"
        if total_emergency_passed else "N/A"
    )

    return GradeResult(
        score=round(score, 4),
        metrics={
            "throughput_per_step":       round(throughput_per_step, 3),
            "throughput_score":          round(throughput_score,    4),
            "emergency_rate_score":      round(em_rate_score,       4),
            "emergency_delay_score":     round(delay_score,         4),
            "efficiency_score":          round(efficiency_score,    4),
            "total_emergency_passed":    total_emergency_passed,
            "avg_emergency_delay_steps": avg_delay_str,
            "total_collisions":          total_collisions,
        },
        feedback=(
            f"Cleared {total_emergency_passed} emergency vehicles "
            f"(avg delay {avg_delay_str}). "
            f"Throughput {throughput_per_step:.2f} veh/step. "
            + ("⚠ Collision!" if total_collisions else "No collisions ✓.")
        ),
    )


# ---------------------------------------------------------------------------
# Task 3 – Dynamic Scenarios  (Hard)
# ---------------------------------------------------------------------------

def _grade_dynamic_scenarios(
    *,
    total_vehicles_passed: int,
    total_emergency_passed: int,
    total_waiting_time: float,
    total_collisions: int,
    total_emergency_delay: float,
    total_phase_changes: int,
    step_count: int,
    **_ignored,
) -> GradeResult:
    throughput_per_step = total_vehicles_passed / step_count
    throughput_score    = min(throughput_per_step / 2.0, 1.0)

    em_rate       = total_emergency_passed / step_count
    em_rate_score = min(em_rate / (1.0 / 15.0), 1.0)

    if total_emergency_passed > 0:
        avg_delay   = total_emergency_delay / total_emergency_passed
        delay_score = max(0.0, 1.0 - avg_delay / 5.0)
    else:
        delay_score = 0.0

    efficiency_score   = 1.0 / (1.0 + total_waiting_time / max(step_count, 1) * 0.08)
    adaptability_score = 1.0 / (1.0 + total_phase_changes / max(step_count, 1) * 0.5)
    collision_penalty  = 0.9 if total_collisions > 0 else 0.0

    raw   = (throughput_score    * 0.25 + em_rate_score   * 0.30 +
             delay_score         * 0.20 + efficiency_score * 0.15 +
             adaptability_score  * 0.10)
    score = max(0.0, raw - collision_penalty)

    return GradeResult(
        score=round(score, 4),
        metrics={
            "throughput_per_step":   round(throughput_per_step, 3),
            "throughput_score":      round(throughput_score,    4),
            "emergency_rate_score":  round(em_rate_score,       4),
            "emergency_delay_score": round(delay_score,         4),
            "efficiency_score":      round(efficiency_score,    4),
            "adaptability_score":    round(adaptability_score,  4),
            "total_collisions":      total_collisions,
            "total_phase_changes":   total_phase_changes,
        },
        feedback=(
            f"Dynamic task: throughput {throughput_per_step:.2f} veh/step, "
            f"{total_emergency_passed} emergencies cleared, "
            f"{total_phase_changes} phase changes over {step_count} steps. "
            + ("⚠ Collision!" if total_collisions else "No collisions ✓.")
        ),
    )
