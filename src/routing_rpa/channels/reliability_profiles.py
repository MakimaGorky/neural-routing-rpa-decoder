"""Coordinate reliability channel profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from routing_rpa.channels.awgn import AWGNChannel, CodeParameters


@dataclass(frozen=True)
class CoordinateReliabilityConfig:
    profile: Literal["prefix"] = "prefix"
    prefix_len: int = 0
    noise_multiplier: float = 1.0


class CoordinateReliabilityAWGN(AWGNChannel):
    """AWGN channel with coordinate-dependent noise scale."""

    def __init__(
        self,
        *,
        prefix_len: int,
        noise_multiplier: float,
        profile: Literal["prefix"] = "prefix",
    ) -> None:
        self.config = CoordinateReliabilityConfig(
            profile=profile,
            prefix_len=prefix_len,
            noise_multiplier=noise_multiplier,
        )
        if self.config.profile != "prefix":
            raise NotImplementedError(
                f"Unsupported reliability profile: {self.config.profile!r}"
            )
        if self.config.prefix_len < 0:
            raise ValueError("prefix_len must be non-negative")
        if self.config.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")

    def coordinate_multipliers(
        self,
        n: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.config.prefix_len > n:
            raise ValueError(f"prefix_len={self.config.prefix_len} exceeds n={n}")

        multipliers = torch.ones(n, device=device, dtype=dtype)
        if self.config.prefix_len:
            multipliers[: self.config.prefix_len] = self.config.noise_multiplier
        return multipliers

    def noise_std(
        self,
        snr: float,
        code: CodeParameters,
        *,
        shape: torch.Size | tuple[int, ...],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(shape) == 0:
            raise ValueError("shape must include a coordinate dimension")
        if shape[-1] != code.n:
            raise ValueError(f"last shape dimension must equal code.n={code.n}, got {shape[-1]}")

        base_sigma = self.sigma(snr, code)
        multipliers = self.coordinate_multipliers(code.n, device=device, dtype=dtype)
        return base_sigma * multipliers.reshape((1,) * (len(shape) - 1) + (code.n,))


__all__ = ["CoordinateReliabilityAWGN", "CoordinateReliabilityConfig"]
