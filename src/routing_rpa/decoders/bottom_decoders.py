"""Bottom decoders for projected RPA subproblems."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from routing_rpa.decoders.kernels_order2 import hadamard_decode_order1
from routing_rpa.projections.projection_set import ProjectionSet


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def build_hadamard_matrix(length: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Build a dense Sylvester Hadamard matrix."""
    if not _is_power_of_two(length):
        raise ValueError(f"Hadamard length must be a positive power of two, got {length}")

    matrix = torch.ones(1, 1, dtype=dtype)
    while matrix.shape[0] < length:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    return matrix


class BottomDecoder(nn.Module, ABC):
    """Base interface for projected bottom decoders."""

    @abstractmethod
    def forward(
        self,
        projected_llr: Tensor,
        projections: ProjectionSet,
        *,
        step: int,
    ) -> Tensor:
        raise NotImplementedError


class HadamardOrder1Decoder(BottomDecoder):
    """Dense Hadamard decoder for first-order RM projected words."""

    def __init__(self, length: int, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.length = int(length)
        self.register_buffer(
            "hadamard_matrix",
            build_hadamard_matrix(self.length, dtype=dtype),
        )

    def forward(
        self,
        projected_llr: Tensor,
        projections: ProjectionSet,
        *,
        step: int,
    ) -> Tensor:
        if projected_llr.dim() != 3:
            raise ValueError(
                f"projected_llr must have shape (B, P, n/2), got {projected_llr.shape}"
            )
        if projected_llr.shape[-1] != self.length:
            raise ValueError(
                f"projected_llr last dimension must be {self.length}, "
                f"got {projected_llr.shape[-1]}"
            )
        if projections.n // 2 != self.length:
            raise ValueError(
                f"projections.n // 2 must match decoder length {self.length}, "
                f"got {projections.n // 2}"
            )
        if projected_llr.shape[1] != projections.num_projections:
            raise ValueError(
                "projected_llr projection count must match ProjectionSet: "
                f"{projected_llr.shape[1]} != {projections.num_projections}"
            )
        if self.hadamard_matrix.device != projected_llr.device:
            raise ValueError(
                "HadamardOrder1Decoder buffer and projected_llr must be on the same device: "
                f"{self.hadamard_matrix.device} != {projected_llr.device}"
            )
        if self.hadamard_matrix.dtype != projected_llr.dtype:
            raise ValueError(
                "HadamardOrder1Decoder buffer and projected_llr must have the same dtype: "
                f"{self.hadamard_matrix.dtype} != {projected_llr.dtype}"
            )

        return hadamard_decode_order1(projected_llr, self.hadamard_matrix)


__all__ = [
    "BottomDecoder",
    "HadamardOrder1Decoder",
    "build_hadamard_matrix",
]
