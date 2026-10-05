from __future__ import annotations

import json
from pathlib import Path

from spade_metaworld.policy_improvement import run_week2_experiment


def main() -> None:
    output_dir = Path("results/week2_policy_improvement")
    result = run_week2_experiment(output_dir)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    print(f"\nArtifacts written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
