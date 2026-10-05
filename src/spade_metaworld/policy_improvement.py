from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sqlalchemy import Float, ForeignKey, Integer, String, create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .complexity import complexity
from .models import Constraints, EnvironmentSpec

FloatArray: TypeAlias = npt.NDArray[np.float64]
ValueTable: TypeAlias = dict[int, float]
ActionValueTable: TypeAlias = dict[tuple[int, int], float]


class PolicyBase(DeclarativeBase):
    """Mappings for the authoritative schema in policy_mdp.sql."""


class World(PolicyBase):
    __tablename__ = "world"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    discount: Mapped[float] = mapped_column(Float)


class DesignState(PolicyBase):
    __tablename__ = "design_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    world_id: Mapped[int] = mapped_column(ForeignKey("world.id"))
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    complexity_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    complexity_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    environment_spec_json: Mapped[str | None] = mapped_column(String, nullable=True)
    is_terminal: Mapped[int] = mapped_column(Integer)
    terminal_kind: Mapped[str | None] = mapped_column(String, nullable=True)


class DesignAction(PolicyBase):
    __tablename__ = "design_action"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)


class Transition(PolicyBase):
    __tablename__ = "transition"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    world_id: Mapped[int] = mapped_column(ForeignKey("world.id"))
    state_id: Mapped[int] = mapped_column(ForeignKey("design_state.id"))
    action_id: Mapped[int] = mapped_column(ForeignKey("design_action.id"))
    next_state_id: Mapped[int] = mapped_column(ForeignKey("design_state.id"))
    probability: Mapped[float] = mapped_column(Float)
    reward: Mapped[float] = mapped_column(Float)


class TransitionObservation(PolicyBase):
    __tablename__ = "transition_observation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    world_id: Mapped[int] = mapped_column(ForeignKey("world.id"))
    state_id: Mapped[int] = mapped_column(ForeignKey("design_state.id"))
    action_id: Mapped[int] = mapped_column(ForeignKey("design_action.id"))
    next_state_id: Mapped[int] = mapped_column(ForeignKey("design_state.id"))
    sample_seed: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String)
    passed_validation: Mapped[int] = mapped_column(Integer)
    reward: Mapped[float] = mapped_column(Float)
    environment_spec_json: Mapped[str | None] = mapped_column(String, nullable=True)
    validation_report_json: Mapped[str | None] = mapped_column(String, nullable=True)


class TransitionModelFit(PolicyBase):
    __tablename__ = "transition_model_fit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    world_id: Mapped[int] = mapped_column(ForeignKey("world.id"))
    name: Mapped[str] = mapped_column(String)
    prior_strength: Mapped[float] = mapped_column(Float)
    samples_per_action: Mapped[int] = mapped_column(Integer)
    random_seed: Mapped[int] = mapped_column(Integer)
    observation_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(String, server_default=func.current_timestamp())


class FittedTransition(PolicyBase):
    __tablename__ = "fitted_transition"

    fit_id: Mapped[int] = mapped_column(ForeignKey("transition_model_fit.id"), primary_key=True)
    state_id: Mapped[int] = mapped_column(ForeignKey("design_state.id"), primary_key=True)
    action_id: Mapped[int] = mapped_column(ForeignKey("design_action.id"), primary_key=True)
    next_state_id: Mapped[int] = mapped_column(ForeignKey("design_state.id"), primary_key=True)
    observed_count: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float] = mapped_column(Float)
    reward_mean: Mapped[float] = mapped_column(Float)


class Policy(PolicyBase):
    __tablename__ = "policy"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    world_id: Mapped[int] = mapped_column(ForeignKey("world.id"))
    name: Mapped[str] = mapped_column(String)
    iteration: Mapped[int] = mapped_column(Integer)


class PolicyProbability(PolicyBase):
    __tablename__ = "policy_probability"

    policy_id: Mapped[int] = mapped_column(ForeignKey("policy.id"), primary_key=True)
    state_id: Mapped[int] = mapped_column(ForeignKey("design_state.id"), primary_key=True)
    action_id: Mapped[int] = mapped_column(ForeignKey("design_action.id"), primary_key=True)
    probability: Mapped[float] = mapped_column(Float)


class AlgorithmRun(PolicyBase):
    __tablename__ = "algorithm_run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    world_id: Mapped[int] = mapped_column(ForeignKey("world.id"))
    name: Mapped[str] = mapped_column(String)
    algorithm: Mapped[str] = mapped_column(String)
    tolerance: Mapped[float] = mapped_column(Float)
    max_iterations: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String)
    iterations: Mapped[int] = mapped_column(Integer)


class ValueSnapshot(PolicyBase):
    __tablename__ = "value_snapshot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("algorithm_run.id"))
    policy_id: Mapped[int] = mapped_column(ForeignKey("policy.id"))
    sequence: Mapped[int] = mapped_column(Integer)
    bellman_residual: Mapped[float] = mapped_column(Float)
    changed_policy_states: Mapped[int | None] = mapped_column(Integer, nullable=True)


class StateValue(PolicyBase):
    __tablename__ = "state_value"

    snapshot_id: Mapped[int] = mapped_column(ForeignKey("value_snapshot.id"), primary_key=True)
    state_id: Mapped[int] = mapped_column(ForeignKey("design_state.id"), primary_key=True)
    value: Mapped[float] = mapped_column(Float)


class ActionValue(PolicyBase):
    __tablename__ = "action_value"

    snapshot_id: Mapped[int] = mapped_column(ForeignKey("value_snapshot.id"), primary_key=True)
    state_id: Mapped[int] = mapped_column(ForeignKey("design_state.id"), primary_key=True)
    action_id: Mapped[int] = mapped_column(ForeignKey("design_action.id"), primary_key=True)
    value: Mapped[float] = mapped_column(Float)


@dataclass(frozen=True)
class ExperimentResult:
    iterative_values: dict[str, float]
    exact_values: dict[str, float]
    optimal_values: dict[str, float]
    optimal_policy: dict[str, list[str]]
    iterative_sweeps: int
    max_evaluation_error: float
    policy_iterations: int

    def as_dict(self) -> dict[str, object]:
        return {
            "iterative_values": self.iterative_values,
            "exact_values": self.exact_values,
            "optimal_values": self.optimal_values,
            "optimal_policy": self.optimal_policy,
            "iterative_sweeps": self.iterative_sweeps,
            "max_evaluation_error": self.max_evaluation_error,
            "policy_iterations": self.policy_iterations,
        }


def create_policy_database(url: str = "sqlite+pysqlite:///:memory:") -> Engine:
    """Create a fresh database using the hand-written Week 2 SQL schema."""

    engine = create_engine(url)
    schema = Path(__file__).with_name("policy_mdp.sql").read_text(encoding="utf-8")
    connection = engine.raw_connection()
    try:
        connection.executescript(schema)
        connection.commit()
    finally:
        connection.close()
    return engine


def _template(
    generation: int,
    task_family: str,
    goal_variations: int,
    episode_horizon: int,
    *,
    action_scale: float = 1.0,
    terminate_on_success: bool = False,
) -> EnvironmentSpec:
    return EnvironmentSpec(
        generation=generation,
        task_family=task_family,  # type: ignore[arg-type]
        goal_variations=goal_variations,
        episode_horizon=episode_horizon,
        rollout_steps=min(episode_horizon, 80),
        seed=20260927 + generation,
        terminate_on_success=terminate_on_success,
        constraints=Constraints(action_scale=action_scale),
        rationale=f"Week 2 MDP template for {task_family}",
    )


def seed_designer_mdp(db: Session) -> tuple[World, Policy]:
    """Store the small MDP and a uniform policy in SQL.

    Transition probabilities are an inspectable planning model. They can later be
    replaced by empirical counts from repeated SPADE proposals and MetaWorld checks.
    """

    world = World(
        name="spade_metaworld_curriculum",
        description=(
            "Choose SPADE environment edits until a valid target-complexity "
            "MetaWorld curriculum environment can be accepted."
        ),
        discount=0.95,
    )
    db.add(world)
    db.flush()

    templates = {
        "foundation": _template(0, "reach-v3", 2, 100),
        "varied": _template(1, "reach-v3", 8, 80),
        "intermediate": _template(2, "push-v3", 8, 70),
        "target": _template(
            3,
            "pick-place-v3",
            12,
            60,
            action_scale=0.85,
            terminate_on_success=True,
        ),
    }
    state_specs = [
        ("invalid", "Proposal failed schema, diversity, or MetaWorld validation.", None, None, None),
        ("foundation", "Valid low-complexity reach environment.", 0.0, 30.0, templates["foundation"]),
        ("varied", "Valid reach environment with increased goal variation.", 30.0, 45.0, templates["varied"]),
        ("intermediate", "Valid push curriculum environment.", 45.0, 65.0, templates["intermediate"]),
        ("target", "Valid target-complexity pick-place environment.", 65.0, 1_000.0, templates["target"]),
        ("accepted", "A target environment was accepted into the curriculum.", None, None, None),
        ("rejected", "An environment was accepted before it met the target.", None, None, None),
    ]
    states: dict[str, DesignState] = {}
    for name, description, lower, upper, template in state_specs:
        terminal_kind = name if name in {"accepted", "rejected"} else None
        state = DesignState(
            world_id=world.id,
            name=name,
            description=description,
            complexity_min=lower,
            complexity_max=upper,
            environment_spec_json=(
                None if template is None else template.model_dump_json()
            ),
            is_terminal=int(terminal_kind is not None),
            terminal_kind=terminal_kind,
        )
        db.add(state)
        states[name] = state
    db.flush()

    action_descriptions = {
        "repair": "Use validation feedback to repair an invalid proposal.",
        "increase_variation": "Increase goal variants while retaining the task family.",
        "advance_family": "Move to the next harder MetaWorld task family.",
        "tighten_constraints": "Shorten the horizon or reduce the action scale.",
        "accept": "Stop editing and accept the current environment.",
    }
    actions: dict[str, DesignAction] = {}
    for name, description in action_descriptions.items():
        action = DesignAction(name=name, description=description)
        db.add(action)
        actions[name] = action
    db.flush()

    transitions: dict[tuple[str, str], list[tuple[str, float, float]]] = {
        ("invalid", "repair"): [("foundation", 0.85, -2.0), ("invalid", 0.15, -2.0)],
        ("foundation", "increase_variation"): [("varied", 0.80, -1.0), ("foundation", 0.20, -1.0)],
        ("foundation", "advance_family"): [("intermediate", 0.45, -2.0), ("varied", 0.40, -2.0), ("invalid", 0.15, -2.0)],
        ("foundation", "tighten_constraints"): [("varied", 0.60, -1.5), ("foundation", 0.30, -1.5), ("invalid", 0.10, -1.5)],
        ("foundation", "accept"): [("rejected", 1.0, -8.0)],
        ("varied", "increase_variation"): [("intermediate", 0.10, -1.0), ("varied", 0.85, -1.0), ("invalid", 0.05, -1.0)],
        ("varied", "advance_family"): [("intermediate", 0.80, -2.0), ("foundation", 0.10, -2.0), ("invalid", 0.10, -2.0)],
        ("varied", "tighten_constraints"): [("intermediate", 0.65, -1.5), ("varied", 0.25, -1.5), ("invalid", 0.10, -1.5)],
        ("varied", "accept"): [("rejected", 1.0, -5.0)],
        ("intermediate", "increase_variation"): [("target", 0.45, -1.0), ("intermediate", 0.45, -1.0), ("invalid", 0.10, -1.0)],
        ("intermediate", "advance_family"): [("target", 0.75, -2.0), ("varied", 0.15, -2.0), ("invalid", 0.10, -2.0)],
        ("intermediate", "tighten_constraints"): [("target", 0.65, -1.5), ("intermediate", 0.25, -1.5), ("invalid", 0.10, -1.5)],
        ("intermediate", "accept"): [("rejected", 1.0, -3.0)],
        ("target", "increase_variation"): [("target", 0.85, -1.0), ("invalid", 0.15, -1.0)],
        ("target", "advance_family"): [("target", 0.80, -2.0), ("invalid", 0.20, -2.0)],
        ("target", "tighten_constraints"): [("target", 0.75, -1.5), ("invalid", 0.25, -1.5)],
        ("target", "accept"): [("accepted", 1.0, 15.0)],
    }
    for (state_name, action_name), outcomes in transitions.items():
        for next_state_name, probability, reward in outcomes:
            db.add(
                Transition(
                    world_id=world.id,
                    state_id=states[state_name].id,
                    action_id=actions[action_name].id,
                    next_state_id=states[next_state_name].id,
                    probability=probability,
                    reward=reward,
                )
            )

    policy = Policy(world_id=world.id, name="uniform", iteration=0)
    db.add(policy)
    db.flush()
    for state in states.values():
        if state.is_terminal:
            continue
        available = sorted(
            {
                actions[action_name].id
                for state_name, action_name in transitions
                if state_name == state.name
            }
        )
        probability = 1.0 / len(available)
        for action_id in available:
            db.add(
                PolicyProbability(
                    policy_id=policy.id,
                    state_id=state.id,
                    action_id=action_id,
                    probability=probability,
                )
            )
    db.flush()
    validate_mdp(db, world)
    return world, policy


def classify_environment(spec: EnvironmentSpec | None, passed_validation: bool) -> str:
    """Map a Week 1 proposal into one of the Week 2 finite MDP states."""

    if spec is None or not passed_validation:
        return "invalid"
    score = complexity(spec).total
    if score < 30.0:
        return "foundation"
    if score < 45.0:
        return "varied"
    if score < 65.0:
        return "intermediate"
    return "target"


def validate_mdp(db: Session, world: World) -> None:
    """Reject incomplete probability distributions before running Bellman updates."""

    rows = db.execute(
        select(
            Transition.state_id,
            Transition.action_id,
            func.sum(Transition.probability),
        )
        .where(Transition.world_id == world.id)
        .group_by(Transition.state_id, Transition.action_id)
    ).all()
    if not rows:
        raise ValueError("The MDP has no transitions.")
    for state_id, action_id, total in rows:
        if not np.isclose(float(total), 1.0):
            raise ValueError(
                f"Transition probabilities for state={state_id}, action={action_id} "
                f"sum to {total}, not 1."
            )


def _states(db: Session, world: World, *, terminal: bool | None = None) -> list[DesignState]:
    statement = select(DesignState).where(DesignState.world_id == world.id)
    if terminal is not None:
        statement = statement.where(DesignState.is_terminal == int(terminal))
    return list(db.scalars(statement.order_by(DesignState.id)))


def _policy_probabilities(db: Session, policy: Policy) -> dict[tuple[int, int], float]:
    return {
        (row.state_id, row.action_id): row.probability
        for row in db.scalars(
            select(PolicyProbability).where(PolicyProbability.policy_id == policy.id)
        )
    }


def action_values(
    db: Session, world: World, values: ValueTable
) -> ActionValueTable:
    """One-step Bellman lookahead: q(s,a)=sum p(r+gamma*V(s'))."""

    result: ActionValueTable = {}
    for transition in db.scalars(
        select(Transition).where(Transition.world_id == world.id)
    ):
        key = (transition.state_id, transition.action_id)
        result[key] = result.get(key, 0.0) + transition.probability * (
            transition.reward + world.discount * values[transition.next_state_id]
        )
    return result


def bellman_expectation_backup(
    db: Session, world: World, policy: Policy, values: ValueTable
) -> tuple[ValueTable, ActionValueTable]:
    probabilities = _policy_probabilities(db, policy)
    q_values = action_values(db, world, values)
    updated = {state.id: 0.0 for state in _states(db, world)}
    for (state_id, action_id), probability in probabilities.items():
        updated[state_id] += probability * q_values[(state_id, action_id)]
    return updated, q_values


def _new_run(
    db: Session,
    world: World,
    name: str,
    algorithm: str,
    tolerance: float,
    max_iterations: int,
) -> AlgorithmRun:
    run = AlgorithmRun(
        world_id=world.id,
        name=name,
        algorithm=algorithm,
        tolerance=tolerance,
        max_iterations=max_iterations,
        status="running",
        iterations=0,
    )
    db.add(run)
    db.flush()
    return run


def _store_snapshot(
    db: Session,
    run: AlgorithmRun,
    policy: Policy,
    values: ValueTable,
    q_values: ActionValueTable,
    residual: float,
    *,
    changed_states: int | None = None,
) -> ValueSnapshot:
    last_sequence = db.scalar(
        select(func.max(ValueSnapshot.sequence)).where(ValueSnapshot.run_id == run.id)
    )
    snapshot = ValueSnapshot(
        run_id=run.id,
        policy_id=policy.id,
        sequence=0 if last_sequence is None else int(last_sequence) + 1,
        bellman_residual=residual,
        changed_policy_states=changed_states,
    )
    db.add(snapshot)
    db.flush()
    for state_id, value in values.items():
        db.add(StateValue(snapshot_id=snapshot.id, state_id=state_id, value=value))
    for (state_id, action_id), value in q_values.items():
        db.add(
            ActionValue(
                snapshot_id=snapshot.id,
                state_id=state_id,
                action_id=action_id,
                value=value,
            )
        )
    db.flush()
    return snapshot


def evaluate_policy_iteratively(
    db: Session,
    world: World,
    policy: Policy,
    *,
    run_name: str = "uniform iterative evaluation",
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> tuple[AlgorithmRun, ValueSnapshot]:
    """Synchronous iterative policy evaluation with every sweep stored in SQL."""

    run = _new_run(
        db, world, run_name, "iterative_evaluation", tolerance, max_iterations
    )
    values = {state.id: 0.0 for state in _states(db, world)}
    for iteration in range(1, max_iterations + 1):
        updated, _ = bellman_expectation_backup(db, world, policy, values)
        residual = max(abs(updated[state_id] - values[state_id]) for state_id in values)
        q_values = action_values(db, world, updated)
        snapshot = _store_snapshot(db, run, policy, updated, q_values, residual)
        values = updated
        run.iterations = iteration
        if residual <= tolerance:
            run.status = "converged"
            db.flush()
            return run, snapshot
    run.status = "failed"
    db.flush()
    raise RuntimeError("Policy evaluation exceeded max_iterations.")


def _exact_values(db: Session, world: World, policy: Policy) -> ValueTable:
    active = _states(db, world, terminal=False)
    all_states = _states(db, world)
    index = {state.id: offset for offset, state in enumerate(active)}
    probabilities = _policy_probabilities(db, policy)
    transition_matrix: FloatArray = np.zeros((len(active), len(active)), dtype=np.float64)
    rewards: FloatArray = np.zeros(len(active), dtype=np.float64)
    for transition in db.scalars(
        select(Transition).where(Transition.world_id == world.id)
    ):
        policy_probability = probabilities.get((transition.state_id, transition.action_id), 0.0)
        if policy_probability == 0.0:
            continue
        row = index[transition.state_id]
        weight = policy_probability * transition.probability
        rewards[row] += weight * transition.reward
        if transition.next_state_id in index:
            transition_matrix[row, index[transition.next_state_id]] += weight
    solution = np.linalg.solve(
        np.eye(len(active), dtype=np.float64) - world.discount * transition_matrix,
        rewards,
    )
    values = {state.id: 0.0 for state in all_states}
    values.update({state.id: float(solution[index[state.id]]) for state in active})
    return values


def _bellman_residual(
    db: Session, world: World, policy: Policy, values: ValueTable
) -> float:
    backed_up, _ = bellman_expectation_backup(db, world, policy, values)
    return max(abs(backed_up[state_id] - values[state_id]) for state_id in values)


def evaluate_policy_exactly(
    db: Session,
    world: World,
    policy: Policy,
    *,
    run_name: str = "uniform exact evaluation",
) -> tuple[AlgorithmRun, ValueSnapshot]:
    """Solve (I - gamma P_pi)V = r_pi and persist the result."""

    run = _new_run(db, world, run_name, "exact_evaluation", 1e-12, 1)
    values = _exact_values(db, world, policy)
    q_values = action_values(db, world, values)
    residual = _bellman_residual(db, world, policy, values)
    snapshot = _store_snapshot(db, run, policy, values, q_values, residual)
    run.iterations = 1
    run.status = "converged"
    db.flush()
    return run, snapshot


def _greedy_distribution(
    db: Session,
    world: World,
    q_values: ActionValueTable,
    *,
    tie_tolerance: float = 1e-10,
) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for state in _states(db, world, terminal=False):
        candidates = {
            action_id: value
            for (state_id, action_id), value in q_values.items()
            if state_id == state.id
        }
        best = max(candidates.values())
        winners = [
            action_id
            for action_id, value in candidates.items()
            if abs(value - best) <= tie_tolerance
        ]
        for action_id in candidates:
            result[(state.id, action_id)] = (
                1.0 / len(winners) if action_id in winners else 0.0
            )
    return result


def _changed_state_count(
    old: dict[tuple[int, int], float], new: dict[tuple[int, int], float]
) -> int:
    state_ids = {state_id for state_id, _ in old} | {state_id for state_id, _ in new}
    return sum(
        any(
            not np.isclose(old.get((state_id, action_id), 0.0), probability)
            for (candidate_state, action_id), probability in new.items()
            if candidate_state == state_id
        )
        for state_id in state_ids
    )


def _store_policy(
    db: Session,
    world: World,
    name: str,
    iteration: int,
    distribution: dict[tuple[int, int], float],
) -> Policy:
    policy = Policy(world_id=world.id, name=name, iteration=iteration)
    db.add(policy)
    db.flush()
    for (state_id, action_id), probability in distribution.items():
        db.add(
            PolicyProbability(
                policy_id=policy.id,
                state_id=state_id,
                action_id=action_id,
                probability=probability,
            )
        )
    db.flush()
    return policy


def improve_policy(
    db: Session,
    world: World,
    policy: Policy,
    values: ValueTable,
    *,
    name: str,
) -> tuple[Policy, int]:
    """Create a policy greedy with respect to the supplied state values."""

    greedy = _greedy_distribution(db, world, action_values(db, world, values))
    changed = _changed_state_count(_policy_probabilities(db, policy), greedy)
    return _store_policy(db, world, name, policy.iteration + 1, greedy), changed


def policy_iteration(
    db: Session,
    world: World,
    initial_policy: Policy,
    *,
    run_name: str = "policy iteration",
    policy_name_prefix: str = "policy_iteration",
    max_iterations: int = 100,
) -> tuple[AlgorithmRun, Policy, ValueSnapshot]:
    """Alternate exact policy evaluation and greedy improvement until stable."""

    run = _new_run(db, world, run_name, "policy_iteration", 1e-10, max_iterations)
    policy = initial_policy
    for iteration in range(max_iterations):
        values = _exact_values(db, world, policy)
        q_values = action_values(db, world, values)
        greedy = _greedy_distribution(db, world, q_values)
        changed = _changed_state_count(_policy_probabilities(db, policy), greedy)
        snapshot = _store_snapshot(
            db,
            run,
            policy,
            values,
            q_values,
            _bellman_residual(db, world, policy, values),
            changed_states=changed,
        )
        run.iterations = iteration + 1
        if changed == 0:
            run.status = "converged"
            db.flush()
            return run, policy, snapshot
        policy = _store_policy(
            db,
            world,
            f"{policy_name_prefix}_{iteration + 1}",
            iteration + 1,
            greedy,
        )
    run.status = "failed"
    db.flush()
    raise RuntimeError("Policy iteration exceeded max_iterations.")


def snapshot_values(db: Session, snapshot: ValueSnapshot) -> ValueTable:
    return {
        row.state_id: row.value
        for row in db.scalars(
            select(StateValue).where(StateValue.snapshot_id == snapshot.id)
        )
    }


def named_values(db: Session, world: World, values: ValueTable) -> dict[str, float]:
    return {state.name: values[state.id] for state in _states(db, world)}


def named_policy(db: Session, world: World, policy: Policy) -> dict[str, list[str]]:
    probabilities = _policy_probabilities(db, policy)
    actions = {
        action.id: action.name for action in db.scalars(select(DesignAction))
    }
    result: dict[str, list[str]] = {}
    for state in _states(db, world, terminal=False):
        result[state.name] = sorted(
            actions[action_id]
            for (state_id, action_id), probability in probabilities.items()
            if state_id == state.id and probability > 0.0
        )
    return result


def run_week2_experiment(output_dir: Path) -> ExperimentResult:
    """Run the complete assignment and save a database, summary, and convergence plot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "policy_improvement.sqlite"
    if database_path.exists():
        database_path.unlink()
    engine = create_policy_database(f"sqlite+pysqlite:///{database_path}")
    with Session(engine) as db:
        world, uniform = seed_designer_mdp(db)
        iterative_run, iterative_snapshot = evaluate_policy_iteratively(db, world, uniform)
        _, exact_snapshot = evaluate_policy_exactly(db, world, uniform)
        iteration_run, optimal_policy, optimal_snapshot = policy_iteration(db, world, uniform)

        iterative = snapshot_values(db, iterative_snapshot)
        exact = snapshot_values(db, exact_snapshot)
        optimal = snapshot_values(db, optimal_snapshot)
        result = ExperimentResult(
            iterative_values=named_values(db, world, iterative),
            exact_values=named_values(db, world, exact),
            optimal_values=named_values(db, world, optimal),
            optimal_policy=named_policy(db, world, optimal_policy),
            iterative_sweeps=iterative_run.iterations,
            max_evaluation_error=max(abs(iterative[key] - exact[key]) for key in exact),
            policy_iterations=iteration_run.iterations,
        )

        residual_rows = db.execute(
            select(ValueSnapshot.sequence, ValueSnapshot.bellman_residual)
            .where(ValueSnapshot.run_id == iterative_run.id)
            .order_by(ValueSnapshot.sequence)
        ).all()
        db.commit()

    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.plot([row[0] + 1 for row in residual_rows], [row[1] for row in residual_rows])
    axis.set_yscale("log")
    axis.set_xlabel("Policy-evaluation sweep")
    axis.set_ylabel("Maximum Bellman update")
    axis.set_title("Uniform-policy evaluation convergence")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "evaluation_convergence.png", dpi=160)
    plt.close(figure)

    (output_dir / "summary.json").write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
