"""Decoder output contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DecoderOutput:
    logits: torch.Tensor
    aux: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


__all__ = ["DecoderOutput"]
