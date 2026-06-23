"""Binary linear code representation."""

from __future__ import annotations

import torch

from routing_rpa.codes.gf2 import gf2_matmul, validate_binary_tensor


class LinearCode:
    """A binary linear code with a generator matrix over GF(2)."""

    def __init__(self, generator_matrix: torch.Tensor) -> None:
        if not isinstance(generator_matrix, torch.Tensor):
            raise TypeError("generator_matrix must be a torch.Tensor")
        if generator_matrix.dim() != 2:
            raise ValueError(
                f"generator_matrix must have shape (k, n), got {generator_matrix.shape}"
            )
        if generator_matrix.shape[0] == 0 or generator_matrix.shape[1] == 0:
            raise ValueError("generator_matrix must have positive k and n")

        validate_binary_tensor(generator_matrix, name="generator_matrix")
        self.generator_matrix = generator_matrix.clone()

    @property
    def k(self) -> int:
        return int(self.generator_matrix.shape[0])

    @property
    def n(self) -> int:
        return int(self.generator_matrix.shape[1])

    @property
    def device(self) -> torch.device:
        return self.generator_matrix.device

    @property
    def dtype(self) -> torch.dtype:
        return self.generator_matrix.dtype

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LinearCode":
        target_device = torch.device(device) if device is not None else self.device
        target_dtype = dtype if dtype is not None else self.dtype
        self.generator_matrix = self.generator_matrix.to(
            device=target_device,
            dtype=target_dtype,
        )
        return self

    def encode(self, messages: torch.Tensor) -> torch.Tensor:
        """Encode a batch of binary messages over GF(2)."""
        if not isinstance(messages, torch.Tensor):
            raise TypeError("messages must be a torch.Tensor")
        if messages.dim() != 2:
            raise ValueError(f"messages must have shape (B, k), got {messages.shape}")
        if messages.shape[1] != self.k:
            raise ValueError(
                f"messages must have {self.k} columns, got {messages.shape[1]}"
            )

        validate_binary_tensor(messages, name="messages")
        messages_on_code_device = messages.to(device=self.device)
        return gf2_matmul(
            messages_on_code_device,
            self.generator_matrix,
            out_dtype=self.dtype,
        )


__all__ = ["LinearCode"]
