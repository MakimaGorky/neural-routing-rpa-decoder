"""Training strategies, losses, metrics, and checkpoints."""

from typing import Any

from routing_rpa.training.checkpoints import (
    BlerPrimaryWithBERTieBreakerPolicy,
    BothImproveCheckpointPolicy,
    CheckpointDecision,
    CheckpointPolicy,
    build_checkpoint_metadata,
    build_checkpoint_policy,
)
from routing_rpa.training.losses import LossComponentConfig, LossComposer, LossConfig
from routing_rpa.training.metrics import (
    BinaryMetricCounts,
    compute_binary_metrics,
    compute_metrics_from_logits,
    logits_to_bits,
    merge_metric_counts,
)
from routing_rpa.training.strategies import LayerwiseTraining, TrainingStrategy


_LAZY_TRAINER_EXPORTS = {
    "CheckpointCallback",
    "DiagnosticsConfig",
    "MetricsLogger",
    "OptimizerFactory",
    "SchedulerFactory",
    "TrainEpochRecord",
    "Trainer",
    "TrainerConfig",
    "TrainingRunResult",
}


def __getattr__(name: str) -> Any:
    """Load trainer exports lazily to avoid eval/training import cycles."""
    if name in _LAZY_TRAINER_EXPORTS:
        from routing_rpa.training import trainer as trainer_module

        value = getattr(trainer_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BinaryMetricCounts",
    "BlerPrimaryWithBERTieBreakerPolicy",
    "BothImproveCheckpointPolicy",
    "CheckpointDecision",
    "CheckpointPolicy",
    "LossComponentConfig",
    "LossComposer",
    "LossConfig",
    "LayerwiseTraining",
    "TrainingStrategy",
    "CheckpointCallback",
    "DiagnosticsConfig",
    "MetricsLogger",
    "OptimizerFactory",
    "SchedulerFactory",
    "TrainEpochRecord",
    "Trainer",
    "TrainerConfig",
    "TrainingRunResult",
    "build_checkpoint_metadata",
    "build_checkpoint_policy",
    "compute_binary_metrics",
    "compute_metrics_from_logits",
    "logits_to_bits",
    "merge_metric_counts",
]
