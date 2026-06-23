"""Structured run logging utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from routing_rpa.utils.serialization import dumps_json, write_json


@dataclass(frozen=True)
class RunDirectory:
    root: Path
    config_json: Path
    metrics_jsonl: Path
    summary_json: Path
    checkpoints_dir: Path
    plots_dir: Path
    artifacts_used_json: Path
    report_txt: Path


def create_run_directory(
    base_dir: str | Path,
    experiment_name: str,
    *,
    timestamp: str | None = None,
    config: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    artifacts_used: dict[str, Any] | None = None,
) -> RunDirectory:
    """Create the standard structured run directory layout."""
    if not experiment_name:
        raise ValueError("experiment_name must be non-empty")
    run_timestamp = timestamp or datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    root = Path(base_dir) / experiment_name / run_timestamp
    checkpoints_dir = root / "checkpoints"
    plots_dir = root / "plots"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    plots_dir.mkdir(parents=True, exist_ok=True)

    directory = RunDirectory(
        root=root,
        config_json=root / "config.json",
        metrics_jsonl=root / "metrics.jsonl",
        summary_json=root / "summary.json",
        checkpoints_dir=checkpoints_dir,
        plots_dir=plots_dir,
        artifacts_used_json=root / "artifacts_used.json",
        report_txt=root / "report.txt",
    )
    write_json(directory.config_json, config or {})
    write_json(directory.summary_json, summary or {})
    write_json(directory.artifacts_used_json, artifacts_used or {})
    directory.metrics_jsonl.touch()
    directory.report_txt.touch()
    return directory


class JsonlWriter:
    """Append structured records to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(dumps_json(record) + "\n")

    def read_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not self.path.exists():
            return records
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records


__all__ = [
    "JsonlWriter",
    "RunDirectory",
    "create_run_directory",
]
