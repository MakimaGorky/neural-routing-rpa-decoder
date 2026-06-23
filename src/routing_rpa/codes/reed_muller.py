"""Reed-Muller code wrapper backed by a generator matrix artifact."""

from __future__ import annotations

from math import comb, log2
from pathlib import Path
from typing import Any

import torch

from routing_rpa.codes.gf2 import require_full_row_rank
from routing_rpa.codes.linear import LinearCode


def rm_dimension(m: int, r: int) -> int:
    """Return k = sum_{i=0}^r C(m, i) for RM(m, r)."""
    if m < 0:
        raise ValueError(f"m must be non-negative, got {m}")
    if r < 0 or r > m:
        raise ValueError(f"r must satisfy 0 <= r <= m, got r={r}, m={m}")
    return sum(comb(m, i) for i in range(r + 1))


def _infer_m_from_length(n: int) -> int:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"RM generator length n must be a power of 2, got n={n}")
    return int(log2(n))


def _infer_r_from_dimension(m: int, k: int) -> int:
    for candidate_r in range(m + 1):
        if rm_dimension(m, candidate_r) == k:
            return candidate_r
    raise ValueError(
        f"Cannot infer RM order r: no r satisfies sum_i=0^r C({m}, i) = {k}"
    )


def _load_text_matrix(path: Path, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    rows: list[list[int]] = []
    expected_width: int | None = None

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = [int(value) for value in stripped.split()]
            if expected_width is None:
                expected_width = len(row)
            elif len(row) != expected_width:
                raise ValueError(
                    f"Inconsistent row width in {path} at line {line_number}: "
                    f"expected {expected_width}, got {len(row)}"
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"Generator matrix artifact is empty: {path}")
    return torch.tensor(rows, dtype=dtype, device=device)


class RMCode(LinearCode):
    """Full Reed-Muller code represented by its generator matrix."""

    def __init__(self, m: int, r: int, generator_matrix: torch.Tensor) -> None:
        self.m = int(m)
        self.r = int(r)
        super().__init__(generator_matrix)

        if self.n != 2**self.m:
            raise ValueError(
                f"RM({self.m}, {self.r}) expects n={2**self.m}, got n={self.n}"
            )

        expected_k = rm_dimension(self.m, self.r)
        if self.k != expected_k:
            raise ValueError(
                f"RM({self.m}, {self.r}) expects k={expected_k}, got k={self.k}"
            )

        require_full_row_rank(self.generator_matrix, name="generator_matrix")

    @classmethod
    def from_generator_matrix(
        cls,
        matrix: torch.Tensor,
        *,
        m: int | None = None,
        r: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "RMCode":
        if not isinstance(matrix, torch.Tensor):
            raise TypeError("matrix must be a torch.Tensor")
        if matrix.dim() != 2:
            raise ValueError(f"matrix must have shape (k, n), got {matrix.shape}")

        target_device = torch.device(device) if device is not None else matrix.device
        target_dtype = dtype if dtype is not None else matrix.dtype
        generator = matrix.to(device=target_device, dtype=target_dtype)

        inferred_m = _infer_m_from_length(int(generator.shape[1])) if m is None else int(m)
        inferred_r = _infer_r_from_dimension(inferred_m, int(generator.shape[0])) if r is None else int(r)
        return cls(inferred_m, inferred_r, generator)

    @classmethod
    def from_text_file(
        cls,
        path: str | Path,
        *,
        m: int | None = None,
        r: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> "RMCode":
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Generator matrix artifact not found: {artifact_path}")

        target_device = torch.device(device)
        matrix = _load_text_matrix(artifact_path, dtype=dtype, device=target_device)
        return cls.from_generator_matrix(matrix, m=m, r=r, dtype=dtype, device=target_device)

    @classmethod
    def from_tensor_file(
        cls,
        path: str | Path,
        *,
        m: int | None = None,
        r: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str = "cpu",
    ) -> "RMCode":
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Generator matrix artifact not found: {artifact_path}")

        target_device = torch.device(device)
        matrix: Any = torch.load(artifact_path, map_location=target_device, weights_only=True)
        if not isinstance(matrix, torch.Tensor):
            raise TypeError(f"Expected tensor artifact in {artifact_path}, got {type(matrix)!r}")

        target_dtype = dtype if dtype is not None else matrix.dtype
        return cls.from_generator_matrix(
            matrix,
            m=m,
            r=r,
            dtype=target_dtype,
            device=target_device,
        )

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        m: int | None = None,
        r: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str = "cpu",
    ) -> "RMCode":
        artifact_path = Path(path)
        if artifact_path.suffix.lower() in {".pt", ".pth"}:
            return cls.from_tensor_file(
                artifact_path,
                m=m,
                r=r,
                dtype=dtype,
                device=device,
            )

        return cls.from_text_file(
            artifact_path,
            m=m,
            r=r,
            dtype=dtype if dtype is not None else torch.float32,
            device=device,
        )


__all__ = ["RMCode", "rm_dimension"]
