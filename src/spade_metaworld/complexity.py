from __future__ import annotations

import math
from dataclasses import dataclass

from .models import EnvironmentSpec

FAMILY_RANK = {
    "reach-v3": 0,
    "push-v3": 1,
    "pick-place-v3": 2,
    "door-open-v3": 3,
    "assembly-v3": 4,
}


@dataclass(frozen=True)
class ComplexityBreakdown:
    family: float
    variation: float
    time_pressure: float
    constraints: float

    @property
    def total(self) -> float:
        return round(self.family + self.variation + self.time_pressure + self.constraints, 4)

    def as_dict(self) -> dict[str, float]:
        return {
            "family": self.family,
            "variation": self.variation,
            "time_pressure": self.time_pressure,
            "constraints": self.constraints,
            "total": self.total,
        }


def complexity(spec: EnvironmentSpec) -> ComplexityBreakdown:
    """Deterministic, inspectable difficulty proxy; it does not claim policy difficulty."""
    family = FAMILY_RANK[spec.task_family] * 20.0
    variation = round(math.log2(spec.goal_variations + 1) * 5.0, 4)
    time_pressure = round((500 - spec.episode_horizon) / 25.0, 4)
    constraints = 0.0
    constraints += (1.0 - spec.constraints.action_scale) * 4.0
    constraints += 1.0 if spec.constraints.min_observation_variance > 0 else 0.0
    constraints += 0.5 if spec.terminate_on_success else 0.0
    return ComplexityBreakdown(family, variation, time_pressure, round(constraints, 4))


def assert_strictly_harder(
    candidate: EnvironmentSpec,
    previous: EnvironmentSpec | None,
    minimum_delta: float = 0.01,
) -> tuple[bool, str]:
    if previous is None:
        return True, "initial environment"
    old = complexity(previous).total
    new = complexity(candidate).total
    delta = new - old
    if delta < minimum_delta:
        return False, f"complexity must increase by >= {minimum_delta}; delta={delta:.4f}"
    return True, f"complexity increased {old:.4f} -> {new:.4f} (delta={delta:.4f})"
