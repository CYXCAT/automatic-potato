from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import EnvironmentSpec
from .validation import ValidationReport


@dataclass
class MemoryEntry:
    timestamp: str
    generation: int
    attempt: int
    accepted: bool
    spec: dict[str, Any]
    fingerprint: str
    report: dict[str, Any]


class JsonlMemory:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / "history.jsonl"
        self.entries: list[MemoryEntry] = []

    def append(self, spec: EnvironmentSpec, report: ValidationReport, attempt: int) -> MemoryEntry:
        entry = MemoryEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            generation=spec.generation,
            attempt=attempt,
            accepted=report.passed,
            spec=spec.model_dump(mode="json"),
            fingerprint=spec.fingerprint,
            report=report.as_dict(),
        )
        self.entries.append(entry)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
        return entry

    @property
    def fingerprints(self) -> set[str]:
        return {entry.fingerprint for entry in self.entries if entry.accepted}
