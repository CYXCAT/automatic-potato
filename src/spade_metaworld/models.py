from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TaskFamily = Literal[
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "assembly-v3",
]


class RolloutPolicy(str, Enum):
    RANDOM = "random"
    GOAL_DIRECTED = "goal_directed"


class Constraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_finite_observations: bool = True
    max_abs_observation: float = Field(default=1_000.0, gt=0)
    min_observation_variance: float = Field(default=0.0, ge=0)
    require_goal_visible: bool = True
    action_scale: float = Field(default=1.0, gt=0, le=1.0)


class EnvironmentSpec(BaseModel):
    """Structured output contract for the environment designer."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    generation: int = Field(ge=0)
    task_family: TaskFamily
    goal_variations: int = Field(ge=1, le=50)
    episode_horizon: int = Field(ge=20, le=500)
    rollout_steps: int = Field(ge=1, le=500)
    seed: int = Field(ge=0, le=2**32 - 1)
    terminate_on_success: bool = False
    constraints: Constraints = Field(default_factory=Constraints)
    rationale: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def rollout_fits_horizon(self) -> EnvironmentSpec:
        if self.rollout_steps > self.episode_horizon:
            raise ValueError("rollout_steps must not exceed episode_horizon")
        return self

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(exclude={"rationale", "generation"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
