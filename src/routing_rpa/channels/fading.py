"""AWGN channel with Rayleigh fading."""

from __future__ import annotations

from math import sqrt

import torch

from routing_rpa.channels.awgn import AWGNChannel, CodeParameters


class AWGNFadingChannel(AWGNChannel):
    """Rayleigh fading followed by additive white Gaussian noise."""

    def __init__(self, *, block_fading: bool = False) -> None:
        self.block_fading = block_fading

    def fading_coefficients(
        self,
        shape: torch.Size | tuple[int, ...],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        fading_shape = (shape[0], 1) if self.block_fading else shape
        std = sqrt(0.5)
        real = torch.normal(0.0, std, size=fading_shape, device=device, dtype=dtype)
        imag = torch.normal(0.0, std, size=fading_shape, device=device, dtype=dtype)
        fading = torch.sqrt(real.square() + imag.square())
        return fading.expand(shape) if self.block_fading else fading

    def add_noise(
        self,
        modulated: torch.Tensor,
        snr: float,
        code: CodeParameters,
    ) -> torch.Tensor:
        fading = self.fading_coefficients(
            modulated.shape,
            device=modulated.device,
            dtype=modulated.dtype,
        )
        std = self.noise_std(
            snr,
            code,
            shape=modulated.shape,
            device=modulated.device,
            dtype=modulated.dtype,
        )
        noise = torch.randn(modulated.shape, device=modulated.device, dtype=modulated.dtype)
        return fading * modulated + noise * std


__all__ = ["AWGNFadingChannel"]
