"""End-to-end experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from routing_rpa.eval.evaluator import EvaluationConfig
from routing_rpa.training.trainer import TrainingRunResult
from routing_rpa.utils.logging import JsonlWriter, RunDirectory, create_run_directory
from routing_rpa.utils.serialization import stable_json_hash, write_json

from routing_rpa.experiments.build import build_experiment
from routing_rpa.experiments.config import ExperimentConfig, load_experiment_config


@dataclass(frozen=True)
class ExperimentRunResult:
    run_directory: RunDirectory
    baseline_metrics: dict[str, Any]
    final_metrics: dict[str, Any]
    training_result: TrainingRunResult | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class CheckpointArtifact:
    path: Path
    decision: dict[str, Any]
    layer_id: int | None
    epoch: int | None
    step: int | None
    ber: float
    bler: float


def run_experiment(
    config_or_path: ExperimentConfig | str | Path,
    *,
    timestamp: str | None = None,
) -> ExperimentRunResult:
    """Load/validate config, build components, run eval/train/eval, and write outputs."""
    config = (
        load_experiment_config(config_or_path)
        if isinstance(config_or_path, (str, Path))
        else config_or_path
    )
    config_snapshot = config.to_dict()
    config_hash = stable_json_hash(config_snapshot)
    run_timestamp = timestamp if timestamp is not None else config.logging.timestamp
    run_directory = create_run_directory(
        Path(config.paths.project_root) / config.paths.runs_dir,
        config.logging.experiment_name,
        timestamp=run_timestamp,
        config=config_snapshot,
        summary={},
        artifacts_used={},
    )
    writer = JsonlWriter(run_directory.metrics_jsonl)

    components = build_experiment(
        config,
        run_directory=run_directory,
        config_snapshot_hash=config_hash,
    )
    write_json(run_directory.artifacts_used_json, components.artifacts_used)

    baseline_config = EvaluationConfig(
        phase="baseline",
        num_batches=config.validation.num_batches,
        batch_size=config.validation.batch_size,
        snr=config.validation.snr,
    )
    baseline_metrics = dict(
        components.evaluator.evaluate(components.evaluation_mode, baseline_config)
    )
    writer.write(baseline_metrics)

    training_result = components.trainer.train() if components.trainer is not None else None
    final_eval_model = _prepare_final_eval_model(
        config=config,
        components=components,
        training_result=training_result,
        run_directory=run_directory,
    )

    final_metrics = dict(
        components.evaluator.evaluate(
            components.evaluation_mode,
            components.final_eval_config,
        )
    )
    final_metrics["model_source"] = final_eval_model["source"]
    if "checkpoint_path" in final_eval_model:
        final_metrics["checkpoint_path"] = final_eval_model["checkpoint_path"]
    writer.write(final_metrics)

    summary = {
        "config_hash": config_hash,
        "run_directory": str(run_directory.root),
        "baseline": baseline_metrics,
        "final_eval": final_metrics,
        "final_eval_model": final_eval_model,
        "training_records": 0 if training_result is None else len(training_result.records),
    }
    write_json(run_directory.summary_json, summary)

    return ExperimentRunResult(
        run_directory=run_directory,
        baseline_metrics=baseline_metrics,
        final_metrics=final_metrics,
        training_result=training_result,
        summary=summary,
    )


def _prepare_final_eval_model(
    *,
    config: ExperimentConfig,
    components: Any,
    training_result: TrainingRunResult | None,
    run_directory: RunDirectory,
) -> dict[str, Any]:
    source = config.final_eval.model_source
    if source == "current":
        return {"source": "current"}
    if source != "best_checkpoint":
        raise ValueError(f"Unsupported final_eval.model_source: {source!r}")

    if training_result is None or training_result.last_record is None:
        raise ValueError("final_eval.model_source='best_checkpoint' requires training records")

    target_layer = training_result.last_record.layer_id
    checkpoint = find_best_checkpoint(run_directory.checkpoints_dir, layer_id=target_layer)
    payload = torch.load(
        checkpoint.path,
        map_location=components.device,
        weights_only=True,
    )
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError(f"Checkpoint does not contain model_state_dict: {checkpoint.path}")
    components.model.load_state_dict(payload["model_state_dict"])

    return {
        "source": "best_checkpoint",
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_decision": checkpoint.decision,
        "layer_id": checkpoint.layer_id,
        "epoch": checkpoint.epoch,
        "global_step": checkpoint.step,
        "validation_ber": checkpoint.ber,
        "validation_bler": checkpoint.bler,
    }


def find_best_checkpoint(
    checkpoints_dir: str | Path,
    *,
    layer_id: int | None = None,
) -> CheckpointArtifact:
    """Find the best saved checkpoint by BLER, using BER as tie-breaker."""
    directory = Path(checkpoints_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")

    candidates: list[CheckpointArtifact] = []
    for path in sorted(directory.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            continue
        decision = payload.get("checkpoint_decision")
        if not isinstance(decision, dict):
            continue
        metadata = decision.get("metadata")
        if not isinstance(metadata, dict):
            continue
        candidate_layer = metadata.get("layer_id")
        if layer_id is not None and candidate_layer != layer_id:
            continue
        monitored = metadata.get("monitored_metrics")
        if not isinstance(monitored, dict):
            continue
        if "ber" not in monitored or "bler" not in monitored:
            continue
        candidates.append(
            CheckpointArtifact(
                path=path,
                decision=decision,
                layer_id=int(candidate_layer) if candidate_layer is not None else None,
                epoch=_optional_int(metadata.get("epoch")),
                step=_optional_int(metadata.get("step")),
                ber=float(monitored["ber"]),
                bler=float(monitored["bler"]),
            )
        )

    if not candidates:
        detail = f" for layer {layer_id}" if layer_id is not None else ""
        raise FileNotFoundError(f"No valid checkpoints found in {directory}{detail}")
    return min(
        candidates,
        key=lambda checkpoint: (
            checkpoint.bler,
            checkpoint.ber,
            checkpoint.step if checkpoint.step is not None else -1,
        ),
    )


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


__all__ = [
    "CheckpointArtifact",
    "ExperimentRunResult",
    "find_best_checkpoint",
    "run_experiment",
]
