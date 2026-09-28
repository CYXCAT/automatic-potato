from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

from .designer import DesignerConfig, RuleBasedDesigner
from .memory import JsonlMemory
from .models import EnvironmentSpec, RolloutPolicy
from .validation import EnvironmentValidator


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_experiment(config_path: Path) -> dict[str, Any]:
    raw = load_config(config_path)
    experiment = raw["experiment"]
    designer_raw = raw["designer"]
    validation_raw = raw["validation"]
    output_dir = (config_path.parent.parent / experiment["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    designer = RuleBasedDesigner(
        DesignerConfig(
            task_families=tuple(designer_raw["task_families"]),
            initial_goal_variations=designer_raw["initial_goal_variations"],
            initial_horizon=designer_raw["initial_horizon"],
            horizon_decrement=designer_raw["horizon_decrement"],
            goal_variation_increment=designer_raw["goal_variation_increment"],
            family_every_n_rounds=designer_raw["family_every_n_rounds"],
        ),
        seed=experiment["seed"],
    )
    validator = EnvironmentValidator(
        rollout_episodes=validation_raw["rollout_episodes"],
        rollout_policy=RolloutPolicy(validation_raw["rollout_policy"]),
        require_diversity=validation_raw["require_diversity"],
        minimum_complexity_delta=validation_raw["minimum_complexity_delta"],
    )
    memory = JsonlMemory(output_dir)
    previous: EnvironmentSpec | None = None
    accepted: list[dict[str, Any]] = []
    feedback: list[str] = []

    for generation in range(experiment["rounds"]):
        for attempt in range(experiment["max_proposals_per_round"]):
            candidate = designer.propose(generation, previous, feedback, attempt)
            report = validator.validate(candidate, previous, memory.fingerprints)
            memory.append(candidate, report, attempt)
            print(
                f"round={generation} attempt={attempt} family={candidate.task_family} "
                f"complexity={report.metrics['complexity']['total']:.4f} "
                f"accepted={report.passed}"
            )
            if report.passed:
                previous = candidate
                accepted.append(
                    {
                        "spec": candidate.model_dump(mode="json"),
                        "fingerprint": candidate.fingerprint,
                        "report": report.as_dict(),
                    }
                )
                break
            feedback = report.messages
        else:
            raise RuntimeError(f"no valid environment found for generation {generation}")

    summary = {
        "experiment": experiment["name"],
        "seed": experiment["seed"],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "rounds_requested": experiment["rounds"],
        "rounds_accepted": len(accepted),
        "policy_training": False,
        "rollout_policy": validation_raw["rollout_policy"],
        "accepted": accepted,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )
    return summary
