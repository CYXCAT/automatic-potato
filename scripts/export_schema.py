import json
from pathlib import Path

from spade_metaworld.models import EnvironmentSpec

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    destination = root / "schemas" / "environment_spec.schema.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(EnvironmentSpec.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(destination)
