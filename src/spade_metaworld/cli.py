from __future__ import annotations

import argparse
from pathlib import Path

from .loop import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SPADE-inspired MetaWorld designer loop")
    parser.add_argument("--config", type=Path, default=Path("configs/minimal.yaml"))
    args = parser.parse_args()
    summary = run_experiment(args.config.resolve())
    print(f"completed {summary['rounds_accepted']}/{summary['rounds_requested']} rounds")


if __name__ == "__main__":
    main()
