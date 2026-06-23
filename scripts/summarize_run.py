"""Summarize an experiment run directory for training diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import torch

from routing_rpa.utils.serialization import dumps_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = summarize_run(args.run_dir)
    if args.output is not None:
        write_json(args.output, report)
    print(dumps_json(report))


def summarize_run(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "summary.json")
    metrics = _read_jsonl(run_dir / "metrics.jsonl")

    phase_counts = Counter(record.get("phase") for record in metrics)
    train_layers = sorted(
        {
            int(record["layer_id"])
            for record in metrics
            if record.get("phase") == "train" and "layer_id" in record
        }
    )
    validation_layers = sorted(
        {
            int(record["layer_id"])
            for record in metrics
            if record.get("phase") == "validation" and "layer_id" in record
        }
    )

    return {
        "run_dir": str(run_dir),
        "config": {
            "seed": config.get("seed"),
            "device": config.get("device"),
            "routing": config.get("routing", {}),
            "training": config.get("training", {}),
            "validation": config.get("validation", {}),
            "final_eval": config.get("final_eval", {}),
        },
        "phase_counts": dict(phase_counts),
        "train_layers": train_layers,
        "validation_layers": validation_layers,
        "best_validation_by_layer": _best_validation_by_layer(metrics),
        "checkpoints": _checkpoint_summaries(run_dir / "checkpoints"),
        "baseline": summary.get("baseline"),
        "final_eval": summary.get("final_eval"),
        "final_eval_model": summary.get("final_eval_model"),
    }


def _best_validation_by_layer(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in metrics:
        if record.get("phase") != "validation" or "layer_id" not in record:
            continue
        key = str(record["layer_id"])
        current = best.get(key)
        if current is None or (record["bler"], record["ber"]) < (current["bler"], current["ber"]):
            best[key] = {
                "epoch": record.get("epoch"),
                "global_step": record.get("global_step"),
                "ber": record.get("ber"),
                "bler": record.get("bler"),
                "checkpoint_should_save": record.get("checkpoint_should_save"),
                "checkpoint_reason": record.get("checkpoint_reason"),
            }
    return best


def _checkpoint_summaries(checkpoints_dir: Path) -> list[dict[str, Any]]:
    if not checkpoints_dir.is_dir():
        return []

    summaries: list[dict[str, Any]] = []
    for path in sorted(checkpoints_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        decision = payload.get("checkpoint_decision", {}) if isinstance(payload, dict) else {}
        metadata = decision.get("metadata", {}) if isinstance(decision, dict) else {}
        summaries.append(
            {
                "path": str(path),
                "layer_id": metadata.get("layer_id"),
                "epoch": metadata.get("epoch"),
                "global_step": metadata.get("step"),
                "monitored_metrics": metadata.get("monitored_metrics"),
                "reason": decision.get("reason") if isinstance(decision, dict) else None,
            }
        )
    return summaries


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                records.append(payload)
    return records


if __name__ == "__main__":
    main()
