"""Build runtime experiment components from dataclass configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import torch

from routing_rpa.data.synthetic import SyntheticCodewordStream
from routing_rpa.decoders.modes import DecoderMode
from routing_rpa.eval.evaluator import EvaluationConfig, Evaluator
from routing_rpa.projections.loaders import load_projection_artifact
from routing_rpa.training.losses import LossComposer
from routing_rpa.training.strategies import LayerwiseTraining
from routing_rpa.training.trainer import DiagnosticsConfig, Trainer, TrainerConfig
from routing_rpa.utils.logging import JsonlWriter, RunDirectory

from routing_rpa.experiments.config import EvalConfig, ExperimentConfig
from routing_rpa.experiments.registry import (
    build_bottom_decoder,
    build_channel,
    build_checkpoint_policy_by_name,
    build_code,
    build_decoder_backend,
    build_optimizer_factory,
    build_routing_policy,
)


@dataclass(frozen=True)
class ExperimentComponents:
    config: ExperimentConfig
    device: torch.device
    code: Any
    projections: Any
    channel_train: Any
    stream: SyntheticCodewordStream
    routing_policy: Any
    bottom_decoder: Any
    model: Any
    evaluator: Evaluator
    trainer: Trainer | None
    evaluation_mode: DecoderMode
    validation_config: EvaluationConfig
    final_eval_config: EvaluationConfig
    artifacts_used: dict[str, Any]


def build_experiment(
    config: ExperimentConfig,
    *,
    run_directory: RunDirectory | None = None,
    config_snapshot_hash: str | None = None,
) -> ExperimentComponents:
    """Build all runtime objects needed by a configured experiment."""
    _set_seed(config.seed)
    device = select_device(config.device)
    project_root = Path(config.paths.project_root)

    code_artifact = _resolve_path(project_root, config.code.generator_artifact)
    code = build_code(config.code, artifact_path=code_artifact, device=device)

    projection_artifact = _resolve_path(project_root, config.projections.artifact)
    projections = load_projection_artifact(
        projection_artifact,
        m=config.projections.m if config.projections.m is not None else config.code.m or 10,
        n=config.projections.n,
        device=device,
        expected_count=config.projections.expected_count,
        metadata=config.projections.metadata or None,
    )

    channel_train = build_channel(config.channel_train)
    stream = SyntheticCodewordStream(code, channel_train, device=device)
    routing_policy = build_routing_policy(
        config.routing,
        code_n=code.n,
        projection_count=projections.num_projections,
        num_unfolded_steps=config.decoder.num_unfolded_steps,
    ).to(device)
    bottom_decoder = build_bottom_decoder(
        config.decoder,
        projections,
        dtype=code.dtype,
        device=device,
    )
    model = build_decoder_backend(
        config.decoder,
        code=code,
        projections=projections,
        bottom_decoder=bottom_decoder,
        routing_policy=routing_policy,
    ).to(device)

    evaluator = Evaluator(model, stream)
    evaluation_mode = decoder_mode_from_config(config)
    validation_config = evaluation_config_from_config(config.validation)
    final_eval_config = evaluation_config_from_config(config.final_eval)
    trainer = _build_trainer(
        config,
        model=model,
        stream=stream,
        evaluator=evaluator,
        run_directory=run_directory,
        config_snapshot_hash=config_snapshot_hash,
    )

    artifacts_used = {
        "code": str(code_artifact),
        "projections": str(projection_artifact),
        "projection_metadata": dict(projections.metadata),
    }
    return ExperimentComponents(
        config=config,
        device=device,
        code=code,
        projections=projections,
        channel_train=channel_train,
        stream=stream,
        routing_policy=routing_policy,
        bottom_decoder=bottom_decoder,
        model=model,
        evaluator=evaluator,
        trainer=trainer,
        evaluation_mode=evaluation_mode,
        validation_config=validation_config,
        final_eval_config=final_eval_config,
        artifacts_used=artifacts_used,
    )


def decoder_mode_from_config(config: ExperimentConfig) -> DecoderMode:
    return DecoderMode(
        selection_scope=config.routing.selection_scope,
        execution_mode=config.routing.execution_mode,
        top_k=config.routing.top_k,
        forward_depth_policy="full_cascade",
        frozen_policy=config.training.frozen_policy,
        collect_debug=config.decoder.collect_debug or config.training.collect_debug,
    )


def evaluation_config_from_config(config: EvalConfig) -> EvaluationConfig:
    return EvaluationConfig(
        phase=config.phase,
        num_batches=config.num_batches,
        batch_size=config.batch_size,
        snr=config.snr,
    )


def select_device(name: str) -> torch.device:
    normalized = name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but torch.cuda.is_available() is false")
    if device.type == "cuda" and device.index is None:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return device


def _build_trainer(
    config: ExperimentConfig,
    *,
    model: Any,
    stream: SyntheticCodewordStream,
    evaluator: Evaluator,
    run_directory: RunDirectory | None,
    config_snapshot_hash: str | None,
) -> Trainer | None:
    if not config.training.enabled:
        return None
    if config.training.strategy != "layerwise":
        raise ValueError(f"Unsupported training strategy: {config.training.strategy!r}")

    metrics_logger = JsonlWriter(run_directory.metrics_jsonl) if run_directory is not None else None
    checkpoint_policy = build_checkpoint_policy_by_name(config.training.checkpoint_policy)
    trainer_config = TrainerConfig(
        epochs=config.training.epochs,
        steps_per_epoch=config.training.steps_per_epoch,
        batch_size=config.training.batch_size,
        train_snr=config.channel_train.snr,
        validation=evaluation_config_from_config(config.validation),
        full_cascade_validation=(
            evaluation_config_from_config(config.training.full_cascade_validation)
            if config.training.full_cascade_validation is not None
            else None
        ),
        diagnostics=_diagnostics_config_from_config(config),
        config_snapshot_path=str(run_directory.config_json) if run_directory is not None else None,
        config_snapshot_hash=config_snapshot_hash,
    )

    return Trainer(
        model=model,
        stream=stream,
        strategy=LayerwiseTraining(
            selection_scope=config.routing.selection_scope,
            execution_mode=config.routing.execution_mode,
            top_k=config.routing.top_k,
            forward_depth_policy=config.training.forward_depth_policy,
            frozen_policy=config.training.frozen_policy,
            train_layers=config.training.train_layers,
            collect_debug=config.training.collect_debug,
        ),
        loss_composer=LossComposer(config.training.loss),
        optimizer_factory=build_optimizer_factory(
            config.training.optimizer,
            lr=config.training.lr,
        ),
        config=trainer_config,
        evaluator=evaluator,
        checkpoint_policy=checkpoint_policy,
        metrics_logger=metrics_logger,
        checkpoint_callback=_build_checkpoint_callback(run_directory),
    )


def _diagnostics_config_from_config(config: ExperimentConfig) -> DiagnosticsConfig | None:
    diagnostics = config.training.diagnostics
    if not diagnostics.enabled:
        return None
    return DiagnosticsConfig(
        phase=diagnostics.phase,
        num_batches=diagnostics.num_batches,
        batch_size=diagnostics.batch_size,
        snr=diagnostics.snr,
    )


def _resolve_path(project_root: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def _build_checkpoint_callback(run_directory: RunDirectory | None):
    if run_directory is None:
        return None

    def save_checkpoint(model: Any, decision: Any) -> None:
        metadata = dict(decision.metadata)
        layer_id = metadata.get("layer_id")
        epoch = metadata.get("epoch")
        step = metadata.get("step")
        filename = f"layer-{layer_id}_epoch-{epoch}_step-{step}.pt"
        path = run_directory.checkpoints_dir / filename
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "checkpoint_decision": {
                    "should_save": decision.should_save,
                    "policy_name": decision.policy_name,
                    "reason": decision.reason,
                    "metadata": metadata,
                },
            },
            path,
        )

    return save_checkpoint


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "ExperimentComponents",
    "build_experiment",
    "decoder_mode_from_config",
    "evaluation_config_from_config",
    "select_device",
]
