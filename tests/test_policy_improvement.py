import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from spade_metaworld.models import EnvironmentSpec
from spade_metaworld.policy_improvement import (
    Transition,
    action_values,
    classify_environment,
    create_policy_database,
    evaluate_policy_exactly,
    evaluate_policy_iteratively,
    named_policy,
    policy_iteration,
    seed_designer_mdp,
    snapshot_values,
)


def test_transitions_form_probability_distributions():
    engine = create_policy_database()
    with Session(engine) as db:
        world, _ = seed_designer_mdp(db)
        totals = db.execute(
            select(func.sum(Transition.probability))
            .where(Transition.world_id == world.id)
            .group_by(Transition.state_id, Transition.action_id)
        ).scalars()
        assert all(np.isclose(total, 1.0) for total in totals)


def test_iterative_evaluation_matches_linear_solve():
    engine = create_policy_database()
    with Session(engine) as db:
        world, uniform = seed_designer_mdp(db)
        iterative_run, iterative_snapshot = evaluate_policy_iteratively(db, world, uniform)
        _, exact_snapshot = evaluate_policy_exactly(db, world, uniform)
        iterative = snapshot_values(db, iterative_snapshot)
        exact = snapshot_values(db, exact_snapshot)

        assert iterative_run.status == "converged"
        assert max(abs(iterative[key] - exact[key]) for key in exact) < 1e-8


def test_policy_iteration_accepts_only_target_state():
    engine = create_policy_database()
    with Session(engine) as db:
        world, uniform = seed_designer_mdp(db)
        run, optimal, snapshot = policy_iteration(db, world, uniform)
        policy = named_policy(db, world, optimal)
        values = snapshot_values(db, snapshot)
        q_values = action_values(db, world, values)

        assert run.status == "converged"
        assert policy["invalid"] == ["repair"]
        assert policy["target"] == ["accept"]
        assert all("accept" not in actions for state, actions in policy.items() if state != "target")
        assert q_values


def test_week1_spec_is_classified_by_existing_complexity_function():
    foundation = EnvironmentSpec(
        generation=0,
        task_family="reach-v3",
        goal_variations=2,
        episode_horizon=100,
        rollout_steps=50,
        seed=7,
        rationale="test",
    )
    target = EnvironmentSpec(
        generation=3,
        task_family="pick-place-v3",
        goal_variations=12,
        episode_horizon=60,
        rollout_steps=50,
        seed=8,
        rationale="test",
    )

    assert classify_environment(None, False) == "invalid"
    assert classify_environment(foundation, True) == "foundation"
    assert classify_environment(target, True) == "target"
