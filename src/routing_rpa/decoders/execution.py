"""Projection execution resolver skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from routing_rpa.decoders.selection import ProjectionPlan
from routing_rpa.projections.projection_set import ProjectionSet


@dataclass(frozen=True)
class ProjectionExecution:
    projections: ProjectionSet
    projection_weights: torch.Tensor
    execution_weights: torch.Tensor
    stats: dict[str, Any]

    @property
    def aggregation_weights(self) -> torch.Tensor:
        return self.projection_weights


class ProjectionExecutor:
    """Resolve selected projections before hot projection/decode kernels."""

    def resolve(
        self,
        projections: ProjectionSet,
        plan: ProjectionPlan,
    ) -> ProjectionExecution:
        if plan.candidate_count != projections.num_projections:
            raise ValueError(
                "plan.candidate_count must match ProjectionSet.num_projections: "
                f"{plan.candidate_count} != {projections.num_projections}"
            )

        if plan.execution_mode == "compute_all_mask":
            if plan.aggregation_weights.shape[-1] != projections.num_projections:
                raise ValueError(
                    "compute_all_mask aggregation_weights must retain candidate length "
                    f"{projections.num_projections}, got {plan.aggregation_weights.shape[-1]}"
                )
            if plan.execution_weights.shape[-1] != projections.num_projections:
                raise ValueError(
                    "compute_all_mask execution_weights must retain candidate length "
                    f"{projections.num_projections}, got {plan.execution_weights.shape[-1]}"
                )
            return ProjectionExecution(
                projections=projections,
                projection_weights=plan.aggregation_weights,
                execution_weights=plan.execution_weights,
                stats=self._stats(
                    candidate_count=projections.num_projections,
                    executed_count=projections.num_projections,
                    aggregated_count=plan.aggregated_count,
                    plan=plan,
                ),
            )

        if plan.execution_mode == "compute_selected":
            if plan.selection_scope == "per_sample":
                raise NotImplementedError(
                    "per_sample compute_selected execution requires specialized kernels"
                )
            if plan.selection_scope not in {"static", "global", "batch"}:
                raise ValueError(
                    "compute_selected requires static/global/batch selection for this "
                    f"skeleton, got {plan.selection_scope!r}"
                )
            if plan.selected_indices is None:
                raise ValueError("compute_selected requires selected_indices")
            if plan.selected_indices.dim() != 1:
                raise ValueError(
                    "compute_selected static/global/batch selected_indices must be 1D"
                )

            cached = plan.aux.get("selected_projections")
            if cached is not None:
                if not isinstance(cached, ProjectionSet):
                    raise TypeError("aux['selected_projections'] must be a ProjectionSet")
                selected_projections = cached
            else:
                selected_projections = projections.subset(plan.selected_indices)
            selected_count = selected_projections.num_projections
            if plan.execution_weights.shape[-1] != selected_count:
                raise ValueError(
                    "compute_selected execution_weights length must match selected count: "
                    f"{plan.execution_weights.shape[-1]} != {selected_count}"
                )
            if plan.aggregation_weights.shape[-1] != selected_count:
                raise ValueError(
                    "compute_selected aggregation_weights length must match selected count: "
                    f"{plan.aggregation_weights.shape[-1]} != {selected_count}"
                )

            return ProjectionExecution(
                projections=selected_projections,
                projection_weights=plan.aggregation_weights,
                execution_weights=plan.execution_weights,
                stats=self._stats(
                    candidate_count=projections.num_projections,
                    executed_count=selected_count,
                    aggregated_count=selected_count,
                    plan=plan,
                ),
            )

        raise ValueError(f"Unsupported execution_mode: {plan.execution_mode!r}")

    @staticmethod
    def _stats(
        *,
        candidate_count: int,
        executed_count: int,
        aggregated_count: int,
        plan: ProjectionPlan,
    ) -> dict[str, Any]:
        return {
            "candidate_projections": candidate_count,
            "executed_projections": executed_count,
            "aggregated_projections": aggregated_count,
            "selection_scope": plan.selection_scope,
            "execution_mode": plan.execution_mode,
        }


__all__ = ["ProjectionExecution", "ProjectionExecutor"]
