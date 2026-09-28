import pytest

from spade_metaworld.metaworld_adapter import build_environment, reset_for_next_variation
from spade_metaworld.models import EnvironmentSpec


@pytest.mark.integration
def test_real_metaworld_reset_and_step():
    spec = EnvironmentSpec(
        generation=0,
        task_family="reach-v3",
        goal_variations=2,
        episode_horizon=20,
        rollout_steps=2,
        seed=123,
        rationale="integration smoke test",
    )
    env = build_environment(spec)
    try:
        obs, _ = reset_for_next_variation(env, seed=123)
        assert env.observation_space.contains(obs)
        transition = env.step(env.action_space.sample())
        assert len(transition) == 5
    finally:
        env.close()
