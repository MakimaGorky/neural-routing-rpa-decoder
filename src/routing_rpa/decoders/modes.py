"""Explicit decoder behavior modes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Literal


SelectionScope = Literal["full", "static", "global", "batch", "per_sample"]
ExecutionMode = Literal["compute_all_mask", "compute_selected"]
ForwardDepthPolicy = Literal["full_cascade", "local_depth"]
FrozenPolicy = Literal[
    "frozen_weights",
    "no_grad_frozen",
    "uniform_frozen",
    "skip_future",
]


VALID_SELECTION_SCOPES = {"full", "static", "global", "batch", "per_sample"}
VALID_EXECUTION_MODES = {"compute_all_mask", "compute_selected"}
VALID_FORWARD_DEPTH_POLICIES = {"full_cascade", "local_depth"}
VALID_FROZEN_POLICIES = {
    "frozen_weights",
    "no_grad_frozen",
    "uniform_frozen",
    "skip_future",
}


@dataclass(frozen=True)
class DecoderMode:
    selection_scope: SelectionScope
    execution_mode: ExecutionMode
    top_k: int | None
    forward_depth_policy: ForwardDepthPolicy
    frozen_policy: FrozenPolicy
    target_layer: int | None = None
    collect_debug: bool = False
    channel_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.selection_scope not in VALID_SELECTION_SCOPES:
            raise ValueError(f"invalid selection_scope: {self.selection_scope!r}")
        if self.execution_mode not in VALID_EXECUTION_MODES:
            raise ValueError(f"invalid execution_mode: {self.execution_mode!r}")
        if self.forward_depth_policy not in VALID_FORWARD_DEPTH_POLICIES:
            raise ValueError(
                f"invalid forward_depth_policy: {self.forward_depth_policy!r}"
            )
        if self.frozen_policy not in VALID_FROZEN_POLICIES:
            raise ValueError(f"invalid frozen_policy: {self.frozen_policy!r}")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be positive or None, got {self.top_k}")
        if self.target_layer is not None and self.target_layer < 0:
            raise ValueError(
                f"target_layer must be non-negative or None, got {self.target_layer}"
            )
        if self.forward_depth_policy == "local_depth" and self.target_layer is None:
            raise ValueError("local_depth requires target_layer")
        if not isinstance(self.channel_context, Mapping):
            raise TypeError("channel_context must be a mapping")
        object.__setattr__(
            self,
            "channel_context",
            MappingProxyType(dict(self.channel_context)),
        )

    def resolve_num_steps(self, num_unfolded_steps: int) -> int:
        """Resolve how many unfolded steps should execute under this mode."""
        if num_unfolded_steps <= 0:
            raise ValueError(
                f"num_unfolded_steps must be positive, got {num_unfolded_steps}"
            )
        if self.frozen_policy == "skip_future" and self.target_layer is not None:
            return min(self.target_layer + 1, num_unfolded_steps)
        if self.forward_depth_policy == "full_cascade":
            return num_unfolded_steps

        assert self.target_layer is not None
        return min(self.target_layer + 1, num_unfolded_steps)

    def is_frozen_step(self, step: int) -> bool:
        """Return whether a step is frozen relative to the active target layer."""
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        return self.target_layer is not None and step != self.target_layer

    def uses_uniform_frozen_weights(self, step: int) -> bool:
        """Return whether a frozen step should ignore route weights and use uniform."""
        return self.frozen_policy == "uniform_frozen" and self.is_frozen_step(step)

    def runs_no_grad(self, step: int) -> bool:
        """Return whether a frozen step should execute under torch.no_grad."""
        return self.frozen_policy == "no_grad_frozen" and self.is_frozen_step(step)

    def with_channel_context(self, **updates: Any) -> "DecoderMode":
        context = dict(self.channel_context)
        context.update(updates)
        return replace(self, channel_context=context)


__all__ = [
    "DecoderMode",
    "ExecutionMode",
    "ForwardDepthPolicy",
    "FrozenPolicy",
    "SelectionScope",
]
