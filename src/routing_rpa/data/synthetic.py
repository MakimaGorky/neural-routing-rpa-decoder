"""Synthetic codeword stream for channel experiments."""

from __future__ import annotations

import torch

from routing_rpa.channels.awgn import AWGNChannel
from routing_rpa.codes.linear import LinearCode


class SyntheticCodewordStream:
    """Generate random codewords and channel outputs on demand."""

    def __init__(
        self,
        code: LinearCode,
        channel: AWGNChannel | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.code = code
        self.channel = channel if channel is not None else AWGNChannel()
        self.device = torch.device(device) if device is not None else code.device
        if self.device != code.device:
            raise ValueError(
                "SyntheticCodewordStream device must match code.device; "
                f"got stream device {self.device} and code device {code.device}"
            )

    def next_batch(self, batch_size: int, snr: float) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        messages = torch.randint(
            0,
            2,
            (batch_size, self.code.k),
            device=self.device,
            dtype=torch.int64,
        ).to(dtype=self.code.dtype)
        target_codeword_bits = self.code.encode(messages)
        channel_output = self.channel(target_codeword_bits, snr, self.code)
        return target_codeword_bits, channel_output


__all__ = ["SyntheticCodewordStream"]
