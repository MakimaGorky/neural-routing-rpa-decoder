"""AWGN channel and LLR conversion utilities."""

from __future__ import annotations

from math import sqrt
from typing import Protocol, overload

import torch


class CodeParameters(Protocol):
    n: int
    k: int


def sigma_sqr_for_code(snr: float, code: CodeParameters) -> float:
    """Return legacy AWGN sigma^2 for a code and SNR in dB."""
    if code.n <= 0 or code.k <= 0:
        raise ValueError(f"code.n and code.k must be positive, got n={code.n}, k={code.k}")
    linear_snr = 10 ** (snr / 10)
    return 1.0 / (2.0 * (code.k / code.n) * linear_snr)


def sigma_for_code(snr: float, code: CodeParameters) -> float:
    """Return legacy AWGN sigma for a code and SNR in dB."""
    return sqrt(sigma_sqr_for_code(snr, code))


def bpsk_modulate(codeword_bits: torch.Tensor) -> torch.Tensor:
    """Map binary bits 0/1 to legacy BPSK symbols +1/-1."""
    return 1.0 - 2.0 * codeword_bits.float()


class AWGNLLR:
    """LLR conversion for the real AWGN channel used by the legacy decoder."""

    def sigma_sqr(self, snr: float, code: CodeParameters) -> float:
        return sigma_sqr_for_code(snr, code)

    def __call__(self, received: torch.Tensor, snr: float, code: CodeParameters) -> torch.Tensor:
        return 2.0 * received / self.sigma_sqr(snr, code)


class FixedSnrAWGNLLR:
    """Decoder-compatible AWGN LLR converter with an explicitly fixed SNR."""

    def __init__(self, snr: float, base_llr: AWGNLLR | None = None) -> None:
        self.snr = float(snr)
        self.base_llr = base_llr if base_llr is not None else AWGNLLR()

    @overload
    def __call__(self, received: torch.Tensor, snr: float, code: CodeParameters) -> torch.Tensor:
        ...

    @overload
    def __call__(self, received: torch.Tensor, snr: CodeParameters) -> torch.Tensor:
        ...

    def __call__(
        self,
        received: torch.Tensor,
        snr: float | CodeParameters,
        code: CodeParameters | None = None,
    ) -> torch.Tensor:
        if code is None:
            code = snr
        return self.base_llr(received, self.snr, code)


class AWGNChannel:
    """BPSK + additive white Gaussian noise."""

    def sigma(self, snr: float, code: CodeParameters) -> float:
        return sigma_for_code(snr, code)

    def sigma_sqr(self, snr: float, code: CodeParameters) -> float:
        return sigma_sqr_for_code(snr, code)

    def modulate(self, codeword_bits: torch.Tensor) -> torch.Tensor:
        return bpsk_modulate(codeword_bits)

    def noise_std(
        self,
        snr: float,
        code: CodeParameters,
        *,
        shape: torch.Size | tuple[int, ...],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.tensor(self.sigma(snr, code), device=device, dtype=dtype)

    def add_noise(
        self,
        modulated: torch.Tensor,
        snr: float,
        code: CodeParameters,
    ) -> torch.Tensor:
        std = self.noise_std(
            snr,
            code,
            shape=modulated.shape,
            device=modulated.device,
            dtype=modulated.dtype,
        )
        noise = torch.randn(modulated.shape, device=modulated.device, dtype=modulated.dtype)
        return modulated + noise * std

    def __call__(
        self,
        codeword_bits: torch.Tensor,
        snr: float,
        code: CodeParameters,
    ) -> torch.Tensor:
        return self.add_noise(self.modulate(codeword_bits), snr, code)


__all__ = [
    "AWGNChannel",
    "FixedSnrAWGNLLR",
    "AWGNLLR",
    "bpsk_modulate",
    "sigma_for_code",
    "sigma_sqr_for_code",
]
