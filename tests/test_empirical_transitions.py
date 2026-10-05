import json
from pathlib import Path

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from spade_metaworld.empirical_transitions import (
    fit_empirical_transition_model,
    ingest_week1_history,
)
from spade_metaworld.policy_improvement import (
    DesignAction,
    DesignState,
    FittedTransition,
    TransitionObservation,
    create_policy_database,
    seed_designer_mdp,
)


def test_week1_history_becomes_transition_evidence(tmp_path: Path):
    payload = {
        "accepted": True,
        "spec": {
            "schema_version": "1.0",
            "generation": 0,
            "task_family": "reach-v3",
            "goal_variations": 2,
            "episode_horizon": 80,
            "rollout_steps": 25,
            "seed": 11,
            "terminate_on_success": False,
            "constraints": {
                "require_finite_observations": True,
                "max_abs_observation": 1000.0,
                "min_observation_variance": 0.0,
                "require_goal_visible": True,
                "action_scale": 1.0,
            },
            "rationale": "history fixture",
        },
        "report": {"passed": True, "checks": {}, "messages": [], "metrics": {}},
    }
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    engine = create_policy_database()
    with Session(engine) as db:
        world, _ = seed_designer_mdp(db)
        assert ingest_week1_history(db, world, history_path) == 1
        observation = db.scalars(select(TransitionObservation)).one()
        states = {row.id: row.name for row in db.scalars(select(DesignState))}
        actions = {row.id: row.name for row in db.scalars(select(DesignAction))}
        assert states[observation.state_id] == "invalid"
        assert actions[observation.action_id] == "repair"
        assert states[observation.next_state_id] == "foundation"


def test_fit_blends_counts_with_weak_manual_prior():
    engine = create_policy_database()
    with Session(engine) as db:
        world, _ = seed_designer_mdp(db)
        states = {row.name: row for row in db.scalars(select(DesignState))}
        actions = {row.name: row for row in db.scalars(select(DesignAction))}
        for seed in range(4):
            db.add(
                TransitionObservation(
                    world_id=world.id,
                    state_id=states["invalid"].id,
                    action_id=actions["repair"].id,
                    next_state_id=states["foundation"].id,
                    sample_seed=seed,
                    source="active_sampling",
                    passed_validation=1,
                    reward=-1.0,
                    environment_spec_json=None,
                    validation_report_json=None,
                )
            )
        db.flush()
        fit = fit_empirical_transition_model(
            db, world, prior_strength=1.0, samples_per_action=0, random_seed=1
        )
        rows = list(
            db.scalars(
                select(FittedTransition).where(
                    FittedTransition.fit_id == fit.id,
                    FittedTransition.state_id == states["invalid"].id,
                    FittedTransition.action_id == actions["repair"].id,
                )
            )
        )
        probabilities = {states_by_id(db)[row.next_state_id]: row.probability for row in rows}
        assert np.isclose(sum(probabilities.values()), 1.0)
        assert probabilities["foundation"] > 0.9
        assert probabilities["invalid"] < 0.1
        assert db.scalar(select(func.count()).select_from(FittedTransition)) > 0


def states_by_id(db: Session) -> dict[int, str]:
    return {row.id: row.name for row in db.scalars(select(DesignState))}
