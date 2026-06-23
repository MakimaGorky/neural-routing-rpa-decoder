"""Evaluation loop for explicit decoder modes and canonical metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from routing_rpa.decoders.modes import DecoderMode
from routing_rpa.decoders.outputs import DecoderOutput
from routing_rpa.training.metrics import BinaryMetricCounts, logits_to_bits


class CodewordStream(Protocol):
    def next_batch(self, batch_size: int, snr: float) -> tuple[torch.Tensor, torch.Tensor]:
        ...


class DecoderModel(Protocol):
    training: bool

    def eval(self) -> nn.Module:
        ...

    def train(self, mode: bool = True) -> nn.Module:
        ...

    def __call__(self, channel_output: torch.Tensor, mode: DecoderMode) -> DecoderOutput:
        ...


@dataclass(frozen=True)
class EvaluationConfig:
    phase: str
    num_batches: int
    batch_size: int
    snr: float

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("phase must be non-empty")
        if self.num_batches <= 0:
            raise ValueError(f"num_batches must be positive, got {self.num_batches}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


class Evaluator:
    """Run a decoder in an explicit mode and accumulate canonical metrics."""

    def __init__(self, model: DecoderModel, stream: CodewordStream) -> None:
        self.model = model
        self.stream = stream

    def evaluate(
        self,
        mode: DecoderMode,
        config: EvaluationConfig,
    ) -> dict[str, float | int | str | object]:
        was_training = self.model.training
        self.model.eval()
        counts = BinaryMetricCounts()
        decoder_stats: dict[str, object] = {}

        try:
            with torch.no_grad():
                for _ in range(config.num_batches):
                    target_codeword, channel_output = self.stream.next_batch(
                        config.batch_size,
                        config.snr,
                    )
                    output = self.model(
                        channel_output,
                        mode.with_channel_context(snr=config.snr),
                    )
                    counts.update(logits_to_bits(output.logits), target_codeword)
                    decoder_stats = dict(output.stats)
        finally:
            self.model.train(was_training)

        return {
            "phase": config.phase,
            **counts.compute(),
            **decoder_stats,
        }


__all__ = [
    "CodewordStream",
    "DecoderModel",
    "EvaluationConfig",
    "Evaluator",
]
