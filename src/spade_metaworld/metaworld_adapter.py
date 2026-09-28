from __future__ import annotations

from typing import Any

import numpy as np

from .models import EnvironmentSpec, RolloutPolicy


def build_environment(spec: EnvironmentSpec) -> Any:
    """Build a real MetaWorld MT1 environment without editing MuJoCo XML."""
    try:
        import gymnasium as gym
        import metaworld  # noqa: F401 - import registers MetaWorld environments
    except ImportError as exc:
        raise RuntimeError(
            "MetaWorld is not installed. Run `uv sync --extra dev` with Python 3.10-3.13."
        ) from exc

    return gym.make(
        "Meta-World/MT1",
        env_name=spec.task_family,
        seed=spec.seed,
        num_goals=spec.goal_variations,
        max_episode_steps=spec.episode_horizon,
        terminate_on_success=spec.terminate_on_success,
        task_select="pseudorandom",
    )


def reset_for_next_variation(env: Any, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Select an MT1 goal, then reset through the complete Gym wrapper stack.

    MetaWorld 3.1.1 creates the MT1 observation space while its goal is hidden and
    later makes the task goal-observable. Refreshing the cached space after task
    selection keeps Gymnasium's declared bounds consistent with actual observations.
    """
    sample_tasks = env.get_wrapper_attr("sample_tasks")
    sample_tasks(seed=seed)
    unwrapped = env.unwrapped
    unwrapped.__dict__.pop("sawyer_observation_space", None)
    unwrapped.observation_space = unwrapped.sawyer_observation_space
    return env.reset(seed=seed)


def choose_action(env: Any, observation: np.ndarray, policy: RolloutPolicy, scale: float):
    if policy is RolloutPolicy.RANDOM:
        action = env.action_space.sample()
    else:
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        # MetaWorld goal-observable states expose hand xyz first and desired goal xyz last.
        if observation.size >= 6 and action.size >= 3:
            action[:3] = np.clip((observation[-3:] - observation[:3]) * 5.0, -1.0, 1.0)
    return np.asarray(action, dtype=np.float32) * scale
