"""Runtime representation of one-dimensional projection geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from routing_rpa.projections.validators import validate_projection_fields


@dataclass(frozen=True)
class ProjectionSet:
    m: int
    n: int
    subspace_dim: int
    directions: torch.Tensor
    coset_indices: torch.Tensor
    metadata: dict[str, Any]
    flat_ids1: torch.Tensor
    flat_ids2: torch.Tensor

    def __post_init__(self) -> None:
        validate_projection_fields(
            m=self.m,
            n=self.n,
            subspace_dim=self.subspace_dim,
            directions=self.directions,
            coset_indices=self.coset_indices,
            metadata=self.metadata,
            flat_ids1=self.flat_ids1,
            flat_ids2=self.flat_ids2,
        )

    @classmethod
    def from_coset_indices(
        cls,
        *,
        m: int,
        n: int,
        subspace_dim: int,
        directions: torch.Tensor,
        coset_indices: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> "ProjectionSet":
        cosets = coset_indices.to(dtype=torch.long)
        dirs = directions.to(device=cosets.device, dtype=torch.long)
        return cls(
            m=m,
            n=n,
            subspace_dim=subspace_dim,
            directions=dirs,
            coset_indices=cosets,
            metadata=dict(metadata or {}),
            flat_ids1=cosets[:, 0, :].reshape(-1),
            flat_ids2=cosets[:, 1, :].reshape(-1),
        )

    @classmethod
    def _from_trusted_parts(
        cls,
        *,
        m: int,
        n: int,
        subspace_dim: int,
        directions: torch.Tensor,
        coset_indices: torch.Tensor,
        metadata: dict[str, Any],
        flat_ids1: torch.Tensor,
        flat_ids2: torch.Tensor,
    ) -> "ProjectionSet":
        """Create a ProjectionSet from tensors derived from an already-valid set.

        This bypasses full artifact validation for hot-path static/global subset
        execution. Callers must only pass tensors sliced from a validated
        ProjectionSet.
        """
        instance = object.__new__(cls)
        object.__setattr__(instance, "m", m)
        object.__setattr__(instance, "n", n)
        object.__setattr__(instance, "subspace_dim", subspace_dim)
        object.__setattr__(instance, "directions", directions)
        object.__setattr__(instance, "coset_indices", coset_indices)
        object.__setattr__(instance, "metadata", metadata)
        object.__setattr__(instance, "flat_ids1", flat_ids1)
        object.__setattr__(instance, "flat_ids2", flat_ids2)
        return instance

    @property
    def num_projections(self) -> int:
        return int(self.coset_indices.shape[0])

    def to(self, device: torch.device | str) -> "ProjectionSet":
        target_device = torch.device(device)
        return ProjectionSet(
            m=self.m,
            n=self.n,
            subspace_dim=self.subspace_dim,
            directions=self.directions.to(device=target_device),
            coset_indices=self.coset_indices.to(device=target_device),
            metadata=dict(self.metadata),
            flat_ids1=self.flat_ids1.to(device=target_device),
            flat_ids2=self.flat_ids2.to(device=target_device),
        )

    def subset(self, indices: torch.Tensor) -> "ProjectionSet":
        if not isinstance(indices, torch.Tensor):
            raise TypeError("indices must be a torch.Tensor")
        if indices.dim() != 1:
            raise ValueError(f"indices must be a 1D tensor, got {indices.shape}")
        if indices.numel() == 0:
            raise ValueError("indices must select at least one projection")

        selected = indices.to(device=self.coset_indices.device, dtype=torch.long)
        if torch.unique(selected).numel() != selected.numel():
            raise ValueError("indices must not contain duplicate projections")

        directions = self.directions.index_select(0, selected)
        cosets = self.coset_indices.index_select(0, selected)

        metadata = dict(self.metadata)
        metadata["parent_num_projections"] = self.num_projections
        metadata["num_projections"] = int(cosets.shape[0])

        return ProjectionSet._from_trusted_parts(
            m=self.m,
            n=self.n,
            subspace_dim=self.subspace_dim,
            directions=directions,
            coset_indices=cosets,
            metadata=metadata,
            flat_ids1=cosets[:, 0, :].reshape(-1),
            flat_ids2=cosets[:, 1, :].reshape(-1),
        )


__all__ = ["ProjectionSet"]
