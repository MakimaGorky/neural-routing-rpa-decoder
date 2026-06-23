"""Training orchestration for explicit decoder modes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from dataclasses import replace
from typing import Protocol

import torch
from torch import nn

from routing_rpa.decoders.outputs import DecoderOutput
from routing_rpa.eval.evaluator import CodewordStream, EvaluationConfig, Evaluator
from routing_rpa.training.checkpoints import CheckpointDecision, CheckpointPolicy
from routing_rpa.training.losses import LossComposer
from routing_rpa.training.strategies import TrainingStrategy


class MetricsLogger(Protocol):
    def write(self, record: dict[str, object]) -> None:
        ...


class CheckpointCallback(Protocol):
    def __call__(
        self,
        model: nn.Module,
        decision: CheckpointDecision,
    ) -> None:
        ...


OptimizerFactory = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
SchedulerFactory = Callable[[torch.optim.Optimizer], object]


@dataclass(frozen=True)
class DiagnosticsConfig:
    phase: str = "diagnostics"
    num_batches: int = 1
    batch_size: int | None = None
    snr: float | None = None

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("diagnostics phase must be non-empty")
        if self.num_batches <= 0:
            raise ValueError(f"diagnostics num_batches must be positive, got {self.num_batches}")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(
                f"diagnostics batch_size must be positive or None, got {self.batch_size}"
            )


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int
    steps_per_epoch: int
    batch_size: int
    train_snr: float
    validation: EvaluationConfig | None = None
    full_cascade_validation: EvaluationConfig | None = None
    diagnostics: DiagnosticsConfig | None = None
    scheduler_metric: str | None = None
    reset_checkpoint_per_layer: bool = True
    config_snapshot_path: str | None = None
    config_snapshot_hash: str | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.steps_per_epoch <= 0:
            raise ValueError(f"steps_per_epoch must be positive, got {self.steps_per_epoch}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


@dataclass(frozen=True)
class TrainEpochRecord:
    layer_id: int
    epoch: int
    global_step: int
    train_metrics: dict[str, object]
    validation_metrics: dict[str, object] = field(default_factory=dict)
    full_cascade_validation_metrics: dict[str, object] = field(default_factory=dict)
    diagnostics_metrics: dict[str, object] = field(default_factory=dict)
    checkpoint_decision: CheckpointDecision | None = None


@dataclass(frozen=True)
class TrainingRunResult:
    records: list[TrainEpochRecord]

    @property
    def last_record(self) -> TrainEpochRecord | None:
        return self.records[-1] if self.records else None


class Trainer:
    """Orchestrate stream, explicit modes, loss, optimizer, validation, and logs."""

    def __init__(
        self,
        *,
        model: nn.Module,
        stream: CodewordStream,
        strategy: TrainingStrategy,
        loss_composer: LossComposer,
        optimizer_factory: OptimizerFactory,
        config: TrainerConfig,
        evaluator: Evaluator | None = None,
        checkpoint_policy: CheckpointPolicy | None = None,
        metrics_logger: MetricsLogger | None = None,
        scheduler_factory: SchedulerFactory | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> None:
        self.model = model
        self.stream = stream
        self.strategy = strategy
        self.loss_composer = loss_composer
        self.optimizer_factory = optimizer_factory
        self.config = config
        self.evaluator = evaluator if evaluator is not None else Evaluator(model, stream)
        self.checkpoint_policy = checkpoint_policy
        self.metrics_logger = metrics_logger
        self.scheduler_factory = scheduler_factory
        self.checkpoint_callback = checkpoint_callback

    def train(self) -> TrainingRunResult:
        records: list[TrainEpochRecord] = []
        global_step = 0

        for layer_id in self.strategy.layers_to_train(self.model):
            if self.checkpoint_policy is not None and self.config.reset_checkpoint_per_layer:
                self.checkpoint_policy.reset()

            parameters = list(self.strategy.trainable_parameters(self.model, layer_id))
            parameter_reference = (
                _clone_parameters(parameters)
                if self.config.diagnostics is not None
                else None
            )
            self._log_layer_start(layer_id=layer_id, parameters=parameters)
            optimizer = self.optimizer_factory(parameters)
            scheduler = (
                self.scheduler_factory(optimizer)
                if self.scheduler_factory is not None
                else None
            )

            for epoch in range(self.config.epochs):
                train_metrics, global_step = self._train_epoch(
                    layer_id=layer_id,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer=optimizer,
                    parameters=parameters,
                    parameter_reference=parameter_reference,
                )
                validation_metrics = self._validate(layer_id=layer_id, epoch=epoch, global_step=global_step)
                checkpoint_decision = self._checkpoint(
                    layer_id=layer_id,
                    epoch=epoch,
                    global_step=global_step,
                    validation_metrics=validation_metrics,
                )
                if checkpoint_decision is not None:
                    validation_metrics["checkpoint_should_save"] = checkpoint_decision.should_save
                    validation_metrics["checkpoint_reason"] = checkpoint_decision.reason
                    validation_metrics["checkpoint_policy"] = checkpoint_decision.policy_name
                full_cascade_validation_metrics = self._validate_full_cascade(
                    layer_id=layer_id,
                    epoch=epoch,
                    global_step=global_step,
                )
                diagnostics_metrics = self._diagnose(
                    layer_id=layer_id,
                    epoch=epoch,
                    global_step=global_step,
                )

                record = TrainEpochRecord(
                    layer_id=layer_id,
                    epoch=epoch,
                    global_step=global_step,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    full_cascade_validation_metrics=full_cascade_validation_metrics,
                    diagnostics_metrics=diagnostics_metrics,
                    checkpoint_decision=checkpoint_decision,
                )
                records.append(record)
                self._log_record(record)
                self._step_scheduler(scheduler, record)

        return TrainingRunResult(records=records)

    def _train_epoch(
        self,
        *,
        layer_id: int,
        epoch: int,
        global_step: int,
        optimizer: torch.optim.Optimizer,
        parameters: list[nn.Parameter],
        parameter_reference: list[torch.Tensor] | None,
    ) -> tuple[dict[str, object], int]:
        self.model.train(True)
        mode = self.strategy.mode_for_step(layer_id, phase="train").with_channel_context(
            snr=self.config.train_snr
        )

        loss_sums: dict[str, float] = {}
        latest_stats: dict[str, object] = {}
        grad_norm_sum = 0.0
        grad_norm_count = 0
        for _ in range(self.config.steps_per_epoch):
            target_codeword, channel_output = self.stream.next_batch(
                self.config.batch_size,
                self.config.train_snr,
            )
            optimizer.zero_grad(set_to_none=True)
            output = self.model(channel_output, mode)
            _validate_decoder_output(output)
            loss, loss_scalars = self.loss_composer(output, target_codeword)
            loss.backward()
            if self.config.diagnostics is not None:
                grad_norm_sum += _parameter_grad_norm(parameters)
                grad_norm_count += 1
            optimizer.step()

            for name, value in loss_scalars.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + float(value)
            latest_stats = dict(output.stats)
            global_step += 1

        loss_averages = {
            f"loss/{name}": value / self.config.steps_per_epoch
            for name, value in loss_sums.items()
        }
        train_metrics: dict[str, object] = {
            "phase": "train",
            "layer_id": layer_id,
            "epoch": epoch,
            "global_step": global_step,
            "train_snr": self.config.train_snr,
            "loss": loss_averages.get("loss/total", 0.0),
            **loss_averages,
            **latest_stats,
        }
        if self.config.diagnostics is not None:
            train_metrics["diagnostics/grad_norm"] = (
                grad_norm_sum / grad_norm_count if grad_norm_count else 0.0
            )
            if parameter_reference is not None:
                train_metrics["diagnostics/parameter_delta_norm"] = _parameter_delta_norm(
                    parameters,
                    parameter_reference,
                )
        return train_metrics, global_step

    def _validate(
        self,
        *,
        layer_id: int,
        epoch: int,
        global_step: int,
    ) -> dict[str, object]:
        if self.config.validation is None:
            return {}

        mode = self.strategy.mode_for_step(layer_id, phase="validation")
        metrics = dict(self.evaluator.evaluate(mode, self.config.validation))
        metrics.update(
            {
                "layer_id": layer_id,
                "epoch": epoch,
                "global_step": global_step,
            }
        )
        return metrics

    def _validate_full_cascade(
        self,
        *,
        layer_id: int,
        epoch: int,
        global_step: int,
    ) -> dict[str, object]:
        if self.config.full_cascade_validation is None:
            return {}

        mode = self.strategy.mode_for_step(None, phase=self.config.full_cascade_validation.phase)
        metrics = dict(self.evaluator.evaluate(mode, self.config.full_cascade_validation))
        metrics.update(
            {
                "layer_id": layer_id,
                "epoch": epoch,
                "global_step": global_step,
                "validation_scope": "full_cascade",
            }
        )
        return metrics

    def _diagnose(
        self,
        *,
        layer_id: int,
        epoch: int,
        global_step: int,
    ) -> dict[str, object]:
        if self.config.diagnostics is None:
            return {}

        diagnostics = self.config.diagnostics
        batch_size = diagnostics.batch_size or self.config.batch_size
        snr = diagnostics.snr if diagnostics.snr is not None else self.config.train_snr
        was_training = self.model.training
        self.model.eval()
        mode = replace(
            self.strategy.mode_for_step(layer_id, phase=diagnostics.phase),
            collect_debug=True,
        ).with_channel_context(snr=snr)

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}

        try:
            with torch.no_grad():
                for _ in range(diagnostics.num_batches):
                    _, channel_output = self.stream.next_batch(batch_size, snr)
                    output = self.model(channel_output, mode)
                    _accumulate_router_debug_scalars(output, sums=sums, counts=counts)
        finally:
            self.model.train(was_training)

        metrics: dict[str, object] = {
            "phase": diagnostics.phase,
            "layer_id": layer_id,
            "epoch": epoch,
            "global_step": global_step,
            "diagnostics_batches": diagnostics.num_batches,
            "diagnostics_batch_size": batch_size,
            "diagnostics_snr": snr,
        }
        metrics.update(
            {
                name: value / counts[name]
                for name, value in sums.items()
                if counts.get(name, 0) > 0
            }
        )
        return metrics

    def _checkpoint(
        self,
        *,
        layer_id: int,
        epoch: int,
        global_step: int,
        validation_metrics: dict[str, object],
    ) -> CheckpointDecision | None:
        if self.checkpoint_policy is None or not validation_metrics:
            return None

        decision = self.checkpoint_policy.decide(
            validation_metrics,
            epoch=epoch,
            step=global_step,
            layer_id=layer_id,
            config_snapshot_path=self.config.config_snapshot_path,
            config_snapshot_hash=self.config.config_snapshot_hash,
        )
        if decision.should_save and self.checkpoint_callback is not None:
            self.checkpoint_callback(self.model, decision)
        return decision

    def _log_record(self, record: TrainEpochRecord) -> None:
        self._write_metrics(record.train_metrics)
        if record.validation_metrics:
            self._write_metrics(record.validation_metrics)
        if record.full_cascade_validation_metrics:
            self._write_metrics(record.full_cascade_validation_metrics)
        if record.diagnostics_metrics:
            self._write_metrics(record.diagnostics_metrics)

    def _log_layer_start(self, *, layer_id: int, parameters: list[nn.Parameter]) -> None:
        if self.config.diagnostics is None:
            return

        model_parameters = list(self.model.parameters())
        trainable_count = sum(parameter.numel() for parameter in model_parameters if parameter.requires_grad)
        total_count = sum(parameter.numel() for parameter in model_parameters)
        active_count = sum(parameter.numel() for parameter in parameters)
        self._write_metrics(
            {
                "phase": "train_layer_start",
                "layer_id": layer_id,
                "active_trainable_parameters": active_count,
                "trainable_parameters": trainable_count,
                "frozen_parameters": total_count - trainable_count,
                "total_parameters": total_count,
            }
        )

    def _write_metrics(self, metrics: dict[str, object]) -> None:
        if self.metrics_logger is None:
            return
        self.metrics_logger.write(metrics)

    def _step_scheduler(self, scheduler: object | None, record: TrainEpochRecord) -> None:
        if scheduler is None:
            return
        step = getattr(scheduler, "step", None)
        if not callable(step):
            raise TypeError("scheduler must expose a callable step method")
        if self.config.scheduler_metric is None:
            step()
            return

        metrics = {**record.train_metrics, **record.validation_metrics}
        if self.config.scheduler_metric not in metrics:
            raise KeyError(f"scheduler metric {self.config.scheduler_metric!r} not found")
        step(float(metrics[self.config.scheduler_metric]))


def _validate_decoder_output(output: DecoderOutput) -> None:
    if not isinstance(output, DecoderOutput):
        raise TypeError("model must return DecoderOutput")


def _clone_parameters(parameters: list[nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _parameter_grad_norm(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total += float(torch.sum(grad * grad).item())
    return total**0.5


def _parameter_delta_norm(
    parameters: list[nn.Parameter],
    reference: list[torch.Tensor],
) -> float:
    if len(parameters) != len(reference):
        raise ValueError("parameter reference length mismatch")

    total = 0.0
    for parameter, initial in zip(parameters, reference, strict=True):
        delta = parameter.detach() - initial.to(device=parameter.device)
        total += float(torch.sum(delta * delta).item())
    return total**0.5


def _accumulate_router_debug_scalars(
    output: DecoderOutput,
    *,
    sums: dict[str, float],
    counts: dict[str, int],
) -> None:
    steps = output.aux.get("steps", [])
    if not isinstance(steps, list):
        return

    for step_record in steps:
        if not isinstance(step_record, dict):
            continue
        step = step_record.get("step")
        if not isinstance(step, int):
            continue

        weights = step_record.get("aggregation_weights")
        if isinstance(weights, torch.Tensor):
            _accumulate_tensor_stats(
                weights,
                prefix=f"router/step_{step}/weights",
                sums=sums,
                counts=counts,
            )

        plan_aux = step_record.get("plan_aux")
        if isinstance(plan_aux, dict):
            logits = plan_aux.get("router_logits")
            if isinstance(logits, torch.Tensor):
                _accumulate_tensor_stats(
                    logits,
                    prefix=f"router/step_{step}/logits",
                    sums=sums,
                    counts=counts,
                )
                probabilities = torch.sigmoid(logits)
                entropy = -(
                    probabilities * torch.log(probabilities.clamp_min(1e-12))
                    + (1.0 - probabilities)
                    * torch.log((1.0 - probabilities).clamp_min(1e-12))
                )
                _accumulate_scalar(
                    f"router/step_{step}/entropy_mean",
                    float(entropy.mean().item()),
                    sums=sums,
                    counts=counts,
                )


def _accumulate_tensor_stats(
    tensor: torch.Tensor,
    *,
    prefix: str,
    sums: dict[str, float],
    counts: dict[str, int],
) -> None:
    detached = tensor.detach()
    _accumulate_scalar(f"{prefix}_mean", float(detached.mean().item()), sums=sums, counts=counts)
    _accumulate_scalar(
        f"{prefix}_std",
        float(detached.std(unbiased=False).item()),
        sums=sums,
        counts=counts,
    )
    _accumulate_scalar(f"{prefix}_min", float(detached.min().item()), sums=sums, counts=counts)
    _accumulate_scalar(f"{prefix}_max", float(detached.max().item()), sums=sums, counts=counts)


def _accumulate_scalar(
    name: str,
    value: float,
    *,
    sums: dict[str, float],
    counts: dict[str, int],
) -> None:
    sums[name] = sums.get(name, 0.0) + value
    counts[name] = counts.get(name, 0) + 1


__all__ = [
    "CheckpointCallback",
    "DiagnosticsConfig",
    "MetricsLogger",
    "OptimizerFactory",
    "SchedulerFactory",
    "TrainEpochRecord",
    "Trainer",
    "TrainerConfig",
    "TrainingRunResult",
]
