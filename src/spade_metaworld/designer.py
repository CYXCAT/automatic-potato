from __future__ import annotations

from dataclasses import dataclass

from .models import Constraints, EnvironmentSpec


@dataclass(frozen=True)
class DesignerConfig:
    task_families: tuple[str, ...]
    initial_goal_variations: int
    initial_horizon: int
    horizon_decrement: int
    goal_variation_increment: int
    family_every_n_rounds: int


class RuleBasedDesigner:
    """Deterministic baseline designer with the same structured contract an LLM can use."""

    def __init__(self, config: DesignerConfig, seed: int):
        self.config = config
        self.seed = seed

    def propose(
        self,
        generation: int,
        previous: EnvironmentSpec | None,
        feedback: list[str],
        attempt: int = 0,
    ) -> EnvironmentSpec:
        family_index = min(
            generation // self.config.family_every_n_rounds,
            len(self.config.task_families) - 1,
        )
        horizon = max(
            20,
            self.config.initial_horizon - generation * self.config.horizon_decrement - attempt,
        )
        variations = min(
            50,
            self.config.initial_goal_variations
            + generation * self.config.goal_variation_increment
            + attempt,
        )
        reason = (
            f"generation {generation}: family stage {family_index}, {variations} goal variants, "
            f"horizon {horizon}; prior feedback: {feedback[-1] if feedback else 'none'}"
        )
        return EnvironmentSpec(
            generation=generation,
            task_family=self.config.task_families[family_index],
            goal_variations=variations,
            episode_horizon=horizon,
            rollout_steps=horizon,
            seed=self.seed + generation * 100 + attempt,
            terminate_on_success=generation >= 3,
            constraints=Constraints(
                require_finite_observations=True,
                require_goal_visible=True,
                action_scale=max(0.6, 1.0 - generation * 0.05),
            ),
            rationale=reason,
        )
