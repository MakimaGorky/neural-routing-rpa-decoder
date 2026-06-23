"""Small GF(2) helpers for binary code artifacts."""

from __future__ import annotations

import torch


def is_binary_tensor(tensor: torch.Tensor) -> bool:
    """Return True when all entries are exactly 0 or 1."""
    if not isinstance(tensor, torch.Tensor):
        return False
    if tensor.numel() == 0:
        return True
    return bool(torch.all((tensor == 0) | (tensor == 1)))


def validate_binary_tensor(tensor: torch.Tensor, *, name: str = "tensor") -> None:
    """Raise ValueError if a tensor contains entries outside GF(2)."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not is_binary_tensor(tensor):
        raise ValueError(f"{name} must contain only binary 0/1 entries")


def gf2_matmul(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Matrix multiplication over GF(2) for two rank-2 tensors."""
    if left.dim() != 2 or right.dim() != 2:
        raise ValueError(
            f"GF(2) matmul expects rank-2 tensors, got {left.shape} and {right.shape}"
        )
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            "GF(2) matmul shape mismatch: "
            f"{left.shape} cannot be multiplied by {right.shape}"
        )

    validate_binary_tensor(left, name="left")
    validate_binary_tensor(right, name="right")

    compute_dtype = (
        torch.float64
        if left.dtype == torch.float64 or right.dtype == torch.float64
        else torch.float32
    )
    left_values = left.to(dtype=compute_dtype)
    right_values = right.to(device=left.device, dtype=compute_dtype)
    result = torch.remainder(left_values @ right_values, 2.0).round()
    return result.to(dtype=out_dtype if out_dtype is not None else left.dtype)


def gf2_row_reduce(matrix: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Return reduced row-echelon form over GF(2) and pivot columns.

    The reduction runs on CPU because it is intended for artifact validation,
    not decoder hot-path execution.
    """
    if matrix.dim() != 2:
        raise ValueError(f"GF(2) row reduction expects a matrix, got {matrix.shape}")
    validate_binary_tensor(matrix, name="matrix")

    reduced = matrix.detach().to(device="cpu", dtype=torch.uint8).clone()
    rows, cols = reduced.shape
    pivot_row = 0
    pivot_columns: list[int] = []

    for col in range(cols):
        pivot = None
        for row in range(pivot_row, rows):
            if bool(reduced[row, col]):
                pivot = row
                break

        if pivot is None:
            continue

        if pivot != pivot_row:
            tmp = reduced[pivot_row].clone()
            reduced[pivot_row] = reduced[pivot]
            reduced[pivot] = tmp

        for row in range(rows):
            if row != pivot_row and bool(reduced[row, col]):
                reduced[row] ^= reduced[pivot_row]

        pivot_columns.append(col)
        pivot_row += 1
        if pivot_row == rows:
            break

    return reduced, pivot_columns


def gf2_rank(matrix: torch.Tensor) -> int:
    """Compute matrix rank over GF(2)."""
    _, pivots = gf2_row_reduce(matrix)
    return len(pivots)


def has_full_row_rank(matrix: torch.Tensor) -> bool:
    """Return True if all rows are linearly independent over GF(2)."""
    if matrix.dim() != 2:
        raise ValueError(f"Expected a matrix, got {matrix.shape}")
    return gf2_rank(matrix) == matrix.shape[0]


def require_full_row_rank(matrix: torch.Tensor, *, name: str = "matrix") -> None:
    """Raise ValueError unless the matrix has full row rank over GF(2)."""
    if not has_full_row_rank(matrix):
        raise ValueError(
            f"{name} must have full row rank over GF(2): "
            f"rank={gf2_rank(matrix)}, rows={matrix.shape[0]}"
        )


__all__ = [
    "gf2_matmul",
    "gf2_rank",
    "gf2_row_reduce",
    "has_full_row_rank",
    "is_binary_tensor",
    "require_full_row_rank",
    "validate_binary_tensor",
]
