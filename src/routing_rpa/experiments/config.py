"""Dataclass experiment configuration and JSON helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from routing_rpa.training.losses import LossConfig
from routing_rpa.utils.serialization import to_jsonable, write_json


@dataclass(frozen=True)
class PathsConfig:
    project_root: str = "."
    runs_dir: str = "runs"

    def __post_init__(self) -> None:
        if not self.project_root:
            raise ValueError("project_root must be non-empty")
        if not self.runs_dir:
            raise ValueError("runs_dir must be non-empty")


@dataclass(frozen=True)
class CodeConfig:
    family: str
    generator_artifact: str
    m: int | None = None
    r: int | None = None
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("code.family must be non-empty")
        if not self.generator_artifact:
            raise ValueError("code.generator_artifact must be non-empty")
        if self.m is not None and self.m < 0:
            raise ValueError(f"code.m must be non-negative, got {self.m}")
        if self.r is not None and self.r < 0:
            raise ValueError(f"code.r must be non-negative, got {self.r}")


@dataclass(frozen=True)
class ProjectionConfig:
    artifact: str
    expected_count: int | None = None
    m: int | None = None
    n: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact:
            raise ValueError("projections.artifact must be non-empty")
        if self.expected_count is not None and self.expected_count <= 0:
            raise ValueError(
                f"projections.expected_count must be positive, got {self.expected_count}"
            )


@dataclass(frozen=True)
class DecoderConfig:
    backend: str = "order2_unfolded"
    num_unfolded_steps: int = 1
    bottom_decoder: str = "hadamard_order1"
    debug_checks: bool = False
    collect_debug: bool = False

    def __post_init__(self) -> None:
        if self.num_unfolded_steps <= 0:
            raise ValueError(
                f"decoder.num_unfolded_steps must be positive, got {self.num_unfolded_steps}"
            )


@dataclass(frozen=True)
class RoutingConfig:
    policy: str
    selection_scope: str
    execution_mode: str
    top_k: int | None = None
    router: str | None = None
    hidden_size: int = 256
    use_layer_norm: bool = False
    activation: str = "relu"
    random_seed: int = 0
    selected_indices: Sequence[int] | None = None
    projection_weights: Sequence[float] | None = None

    def __post_init__(self) -> None:
        if not self.policy:
            raise ValueError("routing.policy must be non-empty")
        if not self.selection_scope:
            raise ValueError("routing.selection_scope must be non-empty")
        if not self.execution_mode:
            raise ValueError("routing.execution_mode must be non-empty")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"routing.top_k must be positive or None, got {self.top_k}")
        if self.hidden_size <= 0:
            raise ValueError(f"routing.hidden_size must be positive, got {self.hidden_size}")
        if self.selected_indices is not None:
            object.__setattr__(self, "selected_indices", tuple(int(i) for i in self.selected_indices))
        if self.projection_weights is not None:
            object.__setattr__(
                self,
                "projection_weights",
                tuple(float(w) for w in self.projection_weights),
            )


@dataclass(frozen=True)
class ChannelConfig:
    name: str = "awgn"
    snr: float = 0.0
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("channel name must be non-empty")


@dataclass(frozen=True)
class EvalConfig:
    phase: str
    num_batches: int
    batch_size: int
    snr: float
    model_source: str = "current"

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("eval phase must be non-empty")
        if self.num_batches <= 0:
            raise ValueError(f"eval.num_batches must be positive, got {self.num_batches}")
        if self.batch_size <= 0:
            raise ValueError(f"eval.batch_size must be positive, got {self.batch_size}")
        if self.model_source not in {"current", "best_checkpoint"}:
            raise ValueError(
                "eval.model_source must be 'current' or 'best_checkpoint', "
                f"got {self.model_source!r}"
            )


@dataclass(frozen=True)
class TrainingDiagnosticsConfig:
    enabled: bool = False
    phase: str = "diagnostics"
    num_batches: int = 1
    batch_size: int | None = None
    snr: float | None = None

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("training.diagnostics.phase must be non-empty")
        if self.num_batches <= 0:
            raise ValueError(
                "training.diagnostics.num_batches must be positive, "
                f"got {self.num_batches}"
            )
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(
                "training.diagnostics.batch_size must be positive or null, "
                f"got {self.batch_size}"
            )


@dataclass(frozen=True)
class TrainingConfig:
    enabled: bool = False
    strategy: str = "layerwise"
    epochs: int = 1
    steps_per_epoch: int = 1
    batch_size: int = 1
    optimizer: str = "adam"
    lr: float = 1e-3
    loss: LossConfig = field(
        default_factory=lambda: LossConfig(
            components=[{"name": "bce_logits", "weight": 1.0}]
        )
    )
    checkpoint_policy: str = "bler_primary_with_ber_tiebreaker"
    forward_depth_policy: str = "local_depth"
    frozen_policy: str = "frozen_weights"
    train_layers: Sequence[int] | None = None
    collect_debug: bool = False
    full_cascade_validation: EvalConfig | None = None
    diagnostics: TrainingDiagnosticsConfig = field(default_factory=TrainingDiagnosticsConfig)

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"training.epochs must be positive, got {self.epochs}")
        if self.steps_per_epoch <= 0:
            raise ValueError(
                f"training.steps_per_epoch must be positive, got {self.steps_per_epoch}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"training.batch_size must be positive, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"training.lr must be positive, got {self.lr}")
        if self.train_layers is not None:
            object.__setattr__(self, "train_layers", tuple(int(i) for i in self.train_layers))


@dataclass(frozen=True)
class LoggingConfig:
    experiment_name: str
    timestamp: str | None = None

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ValueError("logging.experiment_name must be non-empty")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    device: str
    paths: PathsConfig
    code: CodeConfig
    projections: ProjectionConfig
    decoder: DecoderConfig
    routing: RoutingConfig
    channel_train: ChannelConfig
    validation: EvalConfig
    final_eval: EvalConfig
    training: TrainingConfig
    logging: LoggingConfig

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentConfig":
        return cls(
            seed=int(payload["seed"]),
            device=str(payload["device"]),
            paths=_coerce_dataclass(PathsConfig, payload["paths"]),
            code=_coerce_dataclass(CodeConfig, payload["code"]),
            projections=_coerce_dataclass(ProjectionConfig, payload["projections"]),
            decoder=_coerce_dataclass(DecoderConfig, payload["decoder"]),
            routing=_coerce_dataclass(RoutingConfig, payload["routing"]),
            channel_train=_coerce_dataclass(ChannelConfig, payload["channel_train"]),
            validation=_coerce_dataclass(EvalConfig, payload["validation"]),
            final_eval=_coerce_dataclass(EvalConfig, payload["final_eval"]),
            training=_coerce_training_config(payload["training"]),
            logging=_coerce_dataclass(LoggingConfig, payload["logging"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def _coerce_dataclass(cls: type[Any], value: Any) -> Any:
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping for {cls.__name__}, got {type(value)!r}")
    return cls(**dict(value))


def _coerce_loss_config(value: Any) -> LossConfig:
    if isinstance(value, LossConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping for LossConfig, got {type(value)!r}")
    components = value.get("components")
    if components is None:
        raise ValueError("loss config requires 'components'")
    return LossConfig(components=tuple(dict(component) for component in components))


def _coerce_training_config(value: Any) -> TrainingConfig:
    if isinstance(value, TrainingConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping for TrainingConfig, got {type(value)!r}")
    payload = dict(value)
    if "loss" in payload:
        payload["loss"] = _coerce_loss_config(payload["loss"])
    if payload.get("full_cascade_validation") is not None:
        payload["full_cascade_validation"] = _coerce_dataclass(
            EvalConfig,
            payload["full_cascade_validation"],
        )
    if "diagnostics" in payload:
        payload["diagnostics"] = _coerce_dataclass(
            TrainingDiagnosticsConfig,
            payload["diagnostics"],
        )
    return TrainingConfig(**payload)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Experiment config file must contain a JSON object: {path}")
    return ExperimentConfig.from_dict(payload)


def save_experiment_config(path: str | Path, config: ExperimentConfig) -> None:
    write_json(path, config.to_dict())


__all__ = [
    "ChannelConfig",
    "CodeConfig",
    "DecoderConfig",
    "EvalConfig",
    "ExperimentConfig",
    "LoggingConfig",
    "PathsConfig",
    "ProjectionConfig",
    "RoutingConfig",
    "TrainingConfig",
    "TrainingDiagnosticsConfig",
    "load_experiment_config",
    "save_experiment_config",
]
