from spade_metaworld.complexity import assert_strictly_harder, complexity
from spade_metaworld.models import EnvironmentSpec


def make_spec(**changes):
    data = {
        "generation": 0,
        "task_family": "reach-v3",
        "goal_variations": 2,
        "episode_horizon": 80,
        "rollout_steps": 80,
        "seed": 7,
        "rationale": "test candidate",
    }
    data.update(changes)
    return EnvironmentSpec(**data)


def test_more_goal_variations_are_harder():
    before = make_spec()
    after = make_spec(generation=1, goal_variations=4)
    assert complexity(after).total > complexity(before).total
    assert assert_strictly_harder(after, before)[0]


def test_shorter_horizon_is_more_time_pressure():
    before = make_spec()
    after = make_spec(generation=1, episode_horizon=60, rollout_steps=60)
    assert complexity(after).time_pressure > complexity(before).time_pressure


def test_equal_candidate_is_rejected():
    before = make_spec()
    after = make_spec(generation=1)
    passed, message = assert_strictly_harder(after, before)
    assert not passed
    assert "complexity must increase" in message
