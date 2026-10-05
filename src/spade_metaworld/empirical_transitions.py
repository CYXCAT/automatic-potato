from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .complexity import FAMILY_RANK, complexity
from .models import Constraints, EnvironmentSpec, RolloutPolicy
from .policy_improvement import (
    DesignAction,
    DesignState,
    FittedTransition,
    Transition,
    TransitionModelFit,
    TransitionObservation,
    World,
    classify_environment,
    create_policy_database,
    named_policy,
    named_values,
    policy_iteration,
    seed_designer_mdp,
    snapshot_values,
    validate_mdp,
)
from .validation import EnvironmentValidator, ValidationReport

FAMILIES_BY_RANK = tuple(
    family for family, _ in sorted(FAMILY_RANK.items(), key=lambda item: item[1])
)


@dataclass(frozen=True)
class EmpiricalExperimentResult:
    history_observations: int
    active_observations: int
    fitted_transition_rows: int
    prior_strength: float
    manual_policy: dict[str, list[str]]
    empirical_policy: dict[str, list[str]]
    empirical_values: dict[str, float]
    policy_changed_states: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "history_observations": self.history_observations,
            "active_observations": self.active_observations,
            "fitted_transition_rows": self.fitted_transition_rows,
            "prior_strength": self.prior_strength,
            "manual_policy": self.manual_policy,
            "empirical_policy": self.empirical_policy,
            "empirical_values": self.empirical_values,
            "policy_changed_states": self.policy_changed_states,
        }


def _lookup(db: Session, model: type[DesignState | DesignAction]) -> dict[str, Any]:
    return {row.name: row for row in db.scalars(select(model))}


def _spec_for_state(state: DesignState) -> EnvironmentSpec | None:
    if state.environment_spec_json is None:
        return None
    return EnvironmentSpec.model_validate_json(state.environment_spec_json)


def _infer_action(previous: EnvironmentSpec | None, current: EnvironmentSpec) -> str:
    if previous is None:
        return "repair"
    if FAMILY_RANK[current.task_family] > FAMILY_RANK[previous.task_family]:
        return "advance_family"
    if current.goal_variations > previous.goal_variations:
        return "increase_variation"
    return "tighten_constraints"


def _editing_reward(
    source_spec: EnvironmentSpec | None,
    candidate: EnvironmentSpec | None,
    action_name: str,
    passed_validation: bool,
) -> float:
    costs = {
        "repair": -2.0,
        "increase_variation": -1.0,
        "advance_family": -2.0,
        "tighten_constraints": -1.5,
    }
    reward = costs[action_name]
    if not passed_validation or candidate is None:
        return reward - 4.0
    old_score = 0.0 if source_spec is None else complexity(source_spec).total
    progress = max(0.0, complexity(candidate).total - old_score)
    return reward + min(progress / 20.0, 2.0)


def _record_observation(
    db: Session,
    world: World,
    source_state: DesignState,
    action: DesignAction,
    next_state: DesignState,
    *,
    sample_seed: int,
    source: str,
    passed_validation: bool,
    reward: float,
    candidate: EnvironmentSpec | None,
    report: dict[str, Any] | None,
) -> TransitionObservation:
    observation = TransitionObservation(
        world_id=world.id,
        state_id=source_state.id,
        action_id=action.id,
        next_state_id=next_state.id,
        sample_seed=sample_seed,
        source=source,
        passed_validation=int(passed_validation),
        reward=reward,
        environment_spec_json=(
            None if candidate is None else candidate.model_dump_json()
        ),
        validation_report_json=None if report is None else json.dumps(report, sort_keys=True),
    )
    db.add(observation)
    db.flush()
    return observation


def ingest_week1_history(db: Session, world: World, history_path: Path) -> int:
    """Convert the append-only Week 1 JSONL history into transition observations."""

    states = _lookup(db, DesignState)
    actions = _lookup(db, DesignAction)
    previous_spec: EnvironmentSpec | None = None
    previous_passed = False
    count = 0
    with history_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            candidate = EnvironmentSpec.model_validate(payload["spec"])
            passed = bool(payload.get("accepted", payload["report"]["passed"]))
            source_name = classify_environment(previous_spec, previous_passed)
            next_name = classify_environment(candidate, passed)
            action_name = _infer_action(previous_spec, candidate)
            reward = _editing_reward(previous_spec, candidate, action_name, passed)
            _record_observation(
                db,
                world,
                states[source_name],
                actions[action_name],
                states[next_name],
                sample_seed=candidate.seed,
                source="week1_history",
                passed_validation=passed,
                reward=reward,
                candidate=candidate,
                report=payload["report"],
            )
            previous_spec = candidate
            previous_passed = passed
            count += 1
    return count


def _mutate_spec(
    source_spec: EnvironmentSpec | None,
    action_name: str,
    sample_seed: int,
) -> EnvironmentSpec:
    rng = np.random.default_rng(sample_seed)
    if source_spec is None:
        return EnvironmentSpec(
            generation=0,
            task_family="reach-v3",
            goal_variations=int(rng.integers(2, 6)),
            episode_horizon=int(rng.integers(80, 121)),
            rollout_steps=25,
            seed=sample_seed,
            constraints=Constraints(action_scale=float(rng.uniform(0.92, 1.0))),
            rationale=f"empirical repair sample {sample_seed}",
        )

    update: dict[str, Any] = {
        "generation": source_spec.generation + 1,
        "seed": sample_seed,
        "rollout_steps": min(source_spec.episode_horizon, 25),
        "rationale": f"empirical {action_name} sample {sample_seed}",
    }
    if action_name == "increase_variation":
        update["goal_variations"] = min(
            50, source_spec.goal_variations + int(rng.integers(1, 6))
        )
    elif action_name == "advance_family":
        next_rank = min(FAMILY_RANK[source_spec.task_family] + 1, len(FAMILIES_BY_RANK) - 1)
        update["task_family"] = FAMILIES_BY_RANK[next_rank]
        update["goal_variations"] = min(
            50, source_spec.goal_variations + int(rng.integers(1, 4))
        )
    elif action_name == "tighten_constraints":
        new_horizon = max(20, source_spec.episode_horizon - int(rng.integers(5, 16)))
        update["episode_horizon"] = new_horizon
        update["rollout_steps"] = min(new_horizon, 25)
        update["constraints"] = source_spec.constraints.model_copy(
            update={
                "action_scale": max(
                    0.6,
                    source_spec.constraints.action_scale - float(rng.uniform(0.02, 0.10)),
                )
            }
        )
    else:
        raise ValueError(f"Cannot mutate an environment with action {action_name!r}.")
    return source_spec.model_copy(update=update)


def collect_active_observations(
    db: Session,
    world: World,
    *,
    samples_per_action: int = 3,
    random_seed: int = 20261002,
    rollout_episodes: int = 1,
) -> int:
    """Execute designer edits and the real Week 1 MetaWorld validator repeatedly."""

    if samples_per_action < 1:
        raise ValueError("samples_per_action must be positive.")
    states = _lookup(db, DesignState)
    actions = _lookup(db, DesignAction)
    state_by_id = {state.id: state for state in states.values()}
    action_by_id = {action.id: action for action in actions.values()}
    available_pairs = list(
        db.execute(
            select(Transition.state_id, Transition.action_id)
            .where(Transition.world_id == world.id)
            .group_by(Transition.state_id, Transition.action_id)
            .order_by(Transition.state_id, Transition.action_id)
        )
    )
    validator = EnvironmentValidator(
        rollout_episodes=rollout_episodes,
        rollout_policy=RolloutPolicy.RANDOM,
        require_diversity=False,
        minimum_complexity_delta=0.01,
    )
    count = 0
    for pair_index, (state_id, action_id) in enumerate(available_pairs):
        source_state = state_by_id[state_id]
        action = action_by_id[action_id]
        source_spec = _spec_for_state(source_state)
        for sample_index in range(samples_per_action):
            sample_seed = random_seed + pair_index * 1_000 + sample_index
            if action.name == "accept":
                next_name = "accepted" if source_state.name == "target" else "rejected"
                reward = 15.0 if next_name == "accepted" else {
                    "foundation": -8.0,
                    "varied": -5.0,
                    "intermediate": -3.0,
                }[source_state.name]
                _record_observation(
                    db,
                    world,
                    source_state,
                    action,
                    states[next_name],
                    sample_seed=sample_seed,
                    source="active_sampling",
                    passed_validation=True,
                    reward=reward,
                    candidate=source_spec,
                    report={"terminal_decision": next_name},
                )
                count += 1
                continue

            candidate = _mutate_spec(source_spec, action.name, sample_seed)
            report: ValidationReport = validator.validate(candidate, source_spec, set())
            next_name = classify_environment(candidate, report.passed)
            reward = _editing_reward(source_spec, candidate, action.name, report.passed)
            _record_observation(
                db,
                world,
                source_state,
                action,
                states[next_name],
                sample_seed=sample_seed,
                source="active_sampling",
                passed_validation=report.passed,
                reward=reward,
                candidate=candidate,
                report=report.as_dict(),
            )
            count += 1
    return count


def fit_empirical_transition_model(
    db: Session,
    world: World,
    *,
    name: str = "history plus active sampling",
    prior_strength: float = 1.0,
    samples_per_action: int = 3,
    random_seed: int = 20261002,
) -> TransitionModelFit:
    """Fit posterior means using a weak Dirichlet prior from the hand model.

    P_hat(s'|s,a) = (n(s,a,s') + alpha * P_0(s'|s,a)) / (N(s,a) + alpha)
    """

    if prior_strength < 0.0:
        raise ValueError("prior_strength must be nonnegative.")
    manual: dict[tuple[int, int, int], tuple[float, float]] = {
        (row.state_id, row.action_id, row.next_state_id): (row.probability, row.reward)
        for row in db.scalars(select(Transition).where(Transition.world_id == world.id))
    }
    observations = list(
        db.scalars(
            select(TransitionObservation).where(TransitionObservation.world_id == world.id)
        )
    )
    counts: dict[tuple[int, int, int], int] = defaultdict(int)
    reward_sums: dict[tuple[int, int, int], float] = defaultdict(float)
    totals: dict[tuple[int, int], int] = defaultdict(int)
    for row in observations:
        key = (row.state_id, row.action_id, row.next_state_id)
        counts[key] += 1
        reward_sums[key] += row.reward
        totals[(row.state_id, row.action_id)] += 1

    pairs = {(state_id, action_id) for state_id, action_id, _ in manual} | set(totals)
    fit = TransitionModelFit(
        world_id=world.id,
        name=name,
        prior_strength=prior_strength,
        samples_per_action=samples_per_action,
        random_seed=random_seed,
        observation_count=len(observations),
    )
    db.add(fit)
    db.flush()

    fitted_rows: list[tuple[int, int, int, int, float, float]] = []
    for state_id, action_id in sorted(pairs):
        outcomes = {
            next_state_id
            for candidate_state, candidate_action, next_state_id in (
                set(manual) | set(counts)
            )
            if candidate_state == state_id and candidate_action == action_id
        }
        denominator = totals[(state_id, action_id)] + prior_strength
        if denominator <= 0.0:
            raise ValueError(f"No evidence or prior for state={state_id}, action={action_id}.")
        for next_state_id in sorted(outcomes):
            key = (state_id, action_id, next_state_id)
            prior_probability, prior_reward = manual.get(key, (0.0, 0.0))
            probability_numerator = counts[key] + prior_strength * prior_probability
            if probability_numerator <= 0.0:
                continue
            probability = probability_numerator / denominator
            reward = (
                reward_sums[key]
                + prior_strength * prior_probability * prior_reward
            ) / probability_numerator
            fitted_rows.append(
                (state_id, action_id, next_state_id, counts[key], probability, reward)
            )

    db.execute(delete(Transition).where(Transition.world_id == world.id))
    for state_id, action_id, next_state_id, count, probability, reward in fitted_rows:
        db.add(
            FittedTransition(
                fit_id=fit.id,
                state_id=state_id,
                action_id=action_id,
                next_state_id=next_state_id,
                observed_count=count,
                probability=probability,
                reward_mean=reward,
            )
        )
        db.add(
            Transition(
                world_id=world.id,
                state_id=state_id,
                action_id=action_id,
                next_state_id=next_state_id,
                probability=probability,
                reward=reward,
            )
        )
    db.flush()
    validate_mdp(db, world)
    return fit


def fitted_model_as_dict(db: Session, fit: TransitionModelFit) -> list[dict[str, object]]:
    states = {row.id: row.name for row in db.scalars(select(DesignState))}
    actions = {row.id: row.name for row in db.scalars(select(DesignAction))}
    return [
        {
            "state": states[row.state_id],
            "action": actions[row.action_id],
            "next_state": states[row.next_state_id],
            "observed_count": row.observed_count,
            "probability": row.probability,
            "reward_mean": row.reward_mean,
        }
        for row in db.scalars(
            select(FittedTransition)
            .where(FittedTransition.fit_id == fit.id)
            .order_by(
                FittedTransition.state_id,
                FittedTransition.action_id,
                FittedTransition.next_state_id,
            )
        )
    ]


def run_empirical_transition_experiment(
    output_dir: Path,
    history_path: Path,
    *,
    samples_per_action: int = 3,
    prior_strength: float = 1.0,
    random_seed: int = 20261002,
) -> EmpiricalExperimentResult:
    """Collect evidence, fit the model, rerun policy iteration, and save artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "empirical_policy_improvement.sqlite"
    if database_path.exists():
        database_path.unlink()
    engine = create_policy_database(f"sqlite+pysqlite:///{database_path}")
    with Session(engine) as db:
        world, uniform = seed_designer_mdp(db)
        _, manual_optimal, _ = policy_iteration(
            db,
            world,
            uniform,
            run_name="manual model policy iteration",
            policy_name_prefix="manual_policy_iteration",
        )
        manual_policy = named_policy(db, world, manual_optimal)

        history_count = ingest_week1_history(db, world, history_path)
        active_count = collect_active_observations(
            db,
            world,
            samples_per_action=samples_per_action,
            random_seed=random_seed,
        )
        fit = fit_empirical_transition_model(
            db,
            world,
            prior_strength=prior_strength,
            samples_per_action=samples_per_action,
            random_seed=random_seed,
        )
        fitted_model = fitted_model_as_dict(db, fit)

        _, empirical_optimal, empirical_snapshot = policy_iteration(
            db,
            world,
            uniform,
            run_name="empirical model policy iteration",
            policy_name_prefix="empirical_policy_iteration",
        )
        empirical_policy = named_policy(db, world, empirical_optimal)
        empirical_values = named_values(db, world, snapshot_values(db, empirical_snapshot))
        changed_states = sorted(
            state
            for state in empirical_policy
            if empirical_policy[state] != manual_policy[state]
        )
        result = EmpiricalExperimentResult(
            history_observations=history_count,
            active_observations=active_count,
            fitted_transition_rows=len(fitted_model),
            prior_strength=prior_strength,
            manual_policy=manual_policy,
            empirical_policy=empirical_policy,
            empirical_values=empirical_values,
            policy_changed_states=changed_states,
        )
        db.commit()

    (output_dir / "empirical_transition_model.json").write_text(
        json.dumps(fitted_model, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )

    labels = sorted({f"{row['state']} -> {row['action']}" for row in fitted_model})
    observation_totals = {
        label: sum(
            int(row["observed_count"])
            for row in fitted_model
            if f"{row['state']} -> {row['action']}" == label
        )
        for label in labels
    }
    figure, axis = plt.subplots(figsize=(9.0, max(6.0, len(labels) * 0.36)))
    axis.barh(labels, [observation_totals[label] for label in labels])
    axis.set_xlabel("Observed transitions")
    axis.set_title("Evidence collected for each state-action pair")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "transition_evidence.png", dpi=160)
    plt.close(figure)
    return result
