"""Projection selection plan contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from routing_rpa.decoders.modes import VALID_EXECUTION_MODES, VALID_SELECTION_SCOPES


@dataclass
class ProjectionPlan:
    candidate_count: int
    selected_indices: torch.Tensor | None
    selection_scope: str
    execution_mode: str
    aggregation_weights: torch.Tensor
    execution_weights: torch.Tensor
    aux: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.candidate_count <= 0:
            raise ValueError(
                f"candidate_count must be positive, got {self.candidate_count}"
            )
        if self.selection_scope not in VALID_SELECTION_SCOPES:
            raise ValueError(f"invalid selection_scope: {self.selection_scope!r}")
        if self.execution_mode not in VALID_EXECUTION_MODES:
            raise ValueError(f"invalid execution_mode: {self.execution_mode!r}")
        if not isinstance(self.aggregation_weights, torch.Tensor):
            raise TypeError("aggregation_weights must be a torch.Tensor")
        if not isinstance(self.execution_weights, torch.Tensor):
            raise TypeError("execution_weights must be a torch.Tensor")
        if self.aggregation_weights.dim() != 2:
            raise ValueError(
                "aggregation_weights must have shape (1, P/K) or (B, P/K), "
                f"got {self.aggregation_weights.shape}"
            )
        if self.execution_weights.dim() != 2:
            raise ValueError(
                "execution_weights must have shape (1, P/K) or (B, P/K), "
                f"got {self.execution_weights.shape}"
            )
        if self.selected_indices is not None:
            if not isinstance(self.selected_indices, torch.Tensor):
                raise TypeError("selected_indices must be a torch.Tensor or None")
            if self.selected_indices.dim() not in {1, 2}:
                raise ValueError(
                    "selected_indices must have shape (K,) or future (B, K), "
                    f"got {self.selected_indices.shape}"
                )

    @property
    def aggregated_count(self) -> int:
        if "aggregated_count" in self.aux:
            aggregated_count = int(self.aux["aggregated_count"])
            if aggregated_count <= 0:
                raise ValueError("aux['aggregated_count'] must be positive")
            return aggregated_count
        return int(self.aggregation_weights.shape[-1])

    @property
    def execution_count(self) -> int:
        return int(self.execution_weights.shape[-1])


__all__ = ["ProjectionPlan"]
