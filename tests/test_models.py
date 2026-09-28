import pytest
from pydantic import ValidationError

from spade_metaworld.models import EnvironmentSpec


def test_unknown_fields_are_forbidden():
    with pytest.raises(ValidationError):
        EnvironmentSpec(
            generation=0,
            task_family="reach-v3",
            goal_variations=2,
            episode_horizon=50,
            rollout_steps=50,
            seed=1,
            rationale="invalid extra field",
            edits_mujoco_xml=True,
        )


def test_rollout_must_fit_horizon():
    with pytest.raises(ValidationError):
        EnvironmentSpec(
            generation=0,
            task_family="reach-v3",
            goal_variations=2,
            episode_horizon=50,
            rollout_steps=51,
            seed=1,
            rationale="invalid horizon",
        )
