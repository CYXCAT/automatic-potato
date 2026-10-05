from __future__ import annotations

import argparse
import json
from pathlib import Path

from spade_metaworld.empirical_transitions import run_empirical_transition_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the Week 2 MDP from Week 1 history and new MetaWorld trials."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("results/example_seed_20260927/history.jsonl"),
    )
    parser.add_argument("--samples-per-action", type=int, default=3)
    parser.add_argument("--prior-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/week2_empirical_model"),
    )
    args = parser.parse_args()
    result = run_empirical_transition_experiment(
        args.output_dir,
        args.history,
        samples_per_action=args.samples_per_action,
        prior_strength=args.prior_strength,
        random_seed=args.seed,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    print(f"\nArtifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
