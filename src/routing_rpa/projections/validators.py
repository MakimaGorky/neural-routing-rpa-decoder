"""Validation helpers for projection artifacts."""

from __future__ import annotations

from typing import Any

import torch


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _require_integer_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{name} must use an integer dtype, got {tensor.dtype}")


def validate_projection_fields(
    *,
    m: int,
    n: int,
    subspace_dim: int,
    directions: torch.Tensor,
    coset_indices: torch.Tensor,
    metadata: dict[str, Any],
    flat_ids1: torch.Tensor,
    flat_ids2: torch.Tensor,
) -> None:
    """Validate one-dimensional projection geometry and cached flat indices."""
    if not isinstance(m, int) or m < 0:
        raise ValueError(f"m must be a non-negative int, got {m!r}")
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"n must be a positive int, got {n!r}")
    if n != 2**m:
        raise ValueError(f"ProjectionSet requires n == 2**m, got n={n}, m={m}")
    if subspace_dim != 1:
        raise ValueError(
            "Only one-dimensional projections are supported by the current backend, "
            f"got subspace_dim={subspace_dim}"
        )
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict")

    _require_integer_tensor(directions, name="directions")
    _require_integer_tensor(coset_indices, name="coset_indices")
    _require_integer_tensor(flat_ids1, name="flat_ids1")
    _require_integer_tensor(flat_ids2, name="flat_ids2")

    if coset_indices.dim() != 3:
        raise ValueError(
            f"coset_indices must have shape (P, 2, n/2), got {coset_indices.shape}"
        )

    projection_count, halves, half_n = coset_indices.shape
    if projection_count <= 0:
        raise ValueError("ProjectionSet must contain at least one projection")
    if halves != 2 or half_n != n // 2:
        raise ValueError(
            f"coset_indices must have shape (P, 2, {n // 2}), got {coset_indices.shape}"
        )
    if directions.dim() < 1 or directions.shape[0] != projection_count:
        raise ValueError(
            "directions first dimension must match number of projections: "
            f"{directions.shape} vs P={projection_count}"
        )

    expected_flat_shape = (projection_count * half_n,)
    if flat_ids1.shape != expected_flat_shape:
        raise ValueError(
            f"flat_ids1 must have shape {expected_flat_shape}, got {flat_ids1.shape}"
        )
    if flat_ids2.shape != expected_flat_shape:
        raise ValueError(
            f"flat_ids2 must have shape {expected_flat_shape}, got {flat_ids2.shape}"
        )

    device = coset_indices.device
    if directions.device != device or flat_ids1.device != device or flat_ids2.device != device:
        raise ValueError("directions, coset_indices, flat_ids1, and flat_ids2 must share device")

    if int(coset_indices.min()) < 0 or int(coset_indices.max()) >= n:
        raise ValueError(f"Projection indices must be in [0, {n})")

    expected = torch.arange(n, device=device, dtype=coset_indices.dtype)
    flattened_rows = coset_indices.reshape(projection_count, n)
    sorted_rows = torch.sort(flattened_rows, dim=1).values
    if not torch.equal(sorted_rows, expected.expand_as(sorted_rows)):
        raise ValueError("Each projection row must cover every coordinate exactly once")

    expected_flat_ids1 = coset_indices[:, 0, :].reshape(-1)
    expected_flat_ids2 = coset_indices[:, 1, :].reshape(-1)
    if not torch.equal(flat_ids1, expected_flat_ids1):
        raise ValueError("flat_ids1 must match coset_indices[:, 0, :].reshape(-1)")
    if not torch.equal(flat_ids2, expected_flat_ids2):
        raise ValueError("flat_ids2 must match coset_indices[:, 1, :].reshape(-1)")


__all__ = ["validate_projection_fields"]
