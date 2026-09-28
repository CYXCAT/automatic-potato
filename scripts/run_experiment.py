from pathlib import Path

from spade_metaworld.loop import run_experiment

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run_experiment(root / "configs" / "minimal.yaml")
    print(f"results: {root / 'results' / 'latest' / 'summary.json'}")
    print(f"accepted rounds: {result['rounds_accepted']}")
