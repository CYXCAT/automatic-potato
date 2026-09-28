from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .complexity import assert_strictly_harder, complexity
from .metaworld_adapter import build_environment, choose_action, reset_for_next_variation
from .models import EnvironmentSpec, RolloutPolicy


@dataclass
class ValidationReport:
    passed: bool = True
    checks: dict[str, bool] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def record(self, name: str, passed: bool, message: str) -> None:
        self.checks[name] = passed
        self.passed = self.passed and passed
        self.messages.append(f"{name}: {message}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class EnvironmentValidator:
    def __init__(
        self,
        rollout_episodes: int,
        rollout_policy: RolloutPolicy,
        require_diversity: bool,
        minimum_complexity_delta: float,
    ):
        self.rollout_episodes = rollout_episodes
        self.rollout_policy = rollout_policy
        self.require_diversity = require_diversity
        self.minimum_complexity_delta = minimum_complexity_delta

    def validate(
        self,
        spec: EnvironmentSpec,
        previous: EnvironmentSpec | None,
        prior_fingerprints: set[str],
    ) -> ValidationReport:
        report = ValidationReport()
        report.metrics["complexity"] = complexity(spec).as_dict()

        harder, message = assert_strictly_harder(spec, previous, self.minimum_complexity_delta)
        report.record("complexity", harder, message)
        diverse = spec.fingerprint not in prior_fingerprints
        report.record(
            "diversity",
            diverse or not self.require_diversity,
            f"fingerprint {spec.fingerprint} is {'new' if diverse else 'repeated'}",
        )
        if not report.passed:
            return report

        env = None
        try:
            env = build_environment(spec)
            report.record("build", True, f"built Meta-World/MT1 {spec.task_family}")
            self._rollout(env, spec, report)
        # The validator is an isolation boundary: any third-party Gym/MuJoCo failure
        # must become structured feedback rather than terminate the designer loop.
        except Exception as exc:  # noqa: BLE001
            report.record("executable", False, f"{type(exc).__name__}: {exc}")
        finally:
            if env is not None:
                env.close()
        return report

    def _rollout(self, env: Any, spec: EnvironmentSpec, report: ValidationReport) -> None:
        rewards: list[float] = []
        variances: list[float] = []
        successes = 0
        observations_checked = 0
        finite = True
        in_space = True
        bounded = True
        goals: list[tuple[float, ...]] = []

        for episode in range(self.rollout_episodes):
            obs, _ = reset_for_next_variation(env, spec.seed + episode)
            episode_obs = [np.asarray(obs, dtype=np.float64)]
            if np.asarray(obs).size >= 3:
                goals.append(tuple(np.round(np.asarray(obs)[-3:], 8)))
            finite &= bool(np.isfinite(obs).all())
            in_space &= bool(env.observation_space.contains(obs))
            for _ in range(spec.rollout_steps):
                action = choose_action(
                    env,
                    np.asarray(obs),
                    self.rollout_policy,
                    spec.constraints.action_scale,
                )
                obs, reward, terminated, truncated, info = env.step(action)
                arr = np.asarray(obs, dtype=np.float64)
                episode_obs.append(arr)
                observations_checked += 1
                finite &= bool(np.isfinite(arr).all() and np.isfinite(reward))
                in_space &= bool(env.observation_space.contains(obs))
                bounded &= bool(np.abs(arr).max() <= spec.constraints.max_abs_observation)
                rewards.append(float(reward))
                successes += int(bool(info.get("success", False)))
                if terminated or truncated:
                    break
            stacked = np.stack(episode_obs)
            variances.append(float(np.var(stacked, axis=0).mean()))

        variance = float(np.mean(variances))
        unique_goals = len(set(goals))
        report.record("finite", finite, f"checked {observations_checked} transitions")
        report.record("observation_space", in_space, "all observations belong to declared space")
        report.record(
            "bounded", bounded, f"absolute observations <= {spec.constraints.max_abs_observation}"
        )
        report.record(
            "variance",
            variance >= spec.constraints.min_observation_variance,
            f"mean trajectory variance={variance:.8f}",
        )
        report.record(
            "rollout", observations_checked > 0, "random/scripted sanity rollout completed"
        )
        expected_unique = min(self.rollout_episodes, spec.goal_variations)
        report.record(
            "goal_variation",
            unique_goals >= expected_unique,
            f"observed {unique_goals}/{expected_unique} expected distinct sampled goals",
        )
        report.metrics.update(
            {
                "episodes": self.rollout_episodes,
                "transitions": observations_checked,
                "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
                "reward_min": float(np.min(rewards)) if rewards else 0.0,
                "reward_max": float(np.max(rewards)) if rewards else 0.0,
                "success_signals": successes,
                "trajectory_variance": variance,
                "unique_goals_observed": unique_goals,
            }
        )
