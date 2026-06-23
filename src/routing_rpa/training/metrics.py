"""Canonical BER/BLER metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _validate_bit_tensors(predicted_bits: torch.Tensor, target_bits: torch.Tensor) -> None:
    if predicted_bits.shape != target_bits.shape:
        raise ValueError(
            f"predicted_bits and target_bits must have the same shape, "
            f"got {predicted_bits.shape} and {target_bits.shape}"
        )
    if predicted_bits.dim() != 2:
        raise ValueError(f"metrics expect shape (B, n), got {predicted_bits.shape}")
    if predicted_bits.numel() == 0:
        raise ValueError("metrics require at least one bit")


def logits_to_bits(logits: torch.Tensor) -> torch.Tensor:
    """Convert decoder logits to hard bit decisions.

    Positive logits correspond to bit 1 in the new decoder contract.
    """
    return (logits > 0).to(dtype=torch.int64)


@dataclass
class BinaryMetricCounts:
    bit_errors: int = 0
    block_errors: int = 0
    total_bits: int = 0
    total_words: int = 0

    @property
    def correct_words(self) -> int:
        return self.total_words - self.block_errors

    def update(self, predicted_bits: torch.Tensor, target_bits: torch.Tensor) -> None:
        _validate_bit_tensors(predicted_bits, target_bits)
        mismatches = predicted_bits.to(dtype=torch.bool) != target_bits.to(dtype=torch.bool)
        self.bit_errors += int(mismatches.sum().item())
        self.block_errors += int(mismatches.any(dim=1).sum().item())
        self.total_bits += int(target_bits.numel())
        self.total_words += int(target_bits.shape[0])

    def as_dict(self) -> dict[str, int]:
        return {
            "correct_words": self.correct_words,
            "total_words": self.total_words,
            "total_bits": self.total_bits,
            "bit_errors": self.bit_errors,
            "block_errors": self.block_errors,
        }

    def rates(self) -> dict[str, float]:
        if self.total_bits == 0 or self.total_words == 0:
            raise ValueError("Cannot compute metrics before any samples are accumulated")
        return {
            "ber": self.bit_errors / self.total_bits,
            "bler": self.block_errors / self.total_words,
        }

    def compute(self) -> dict[str, float | int]:
        return {**self.rates(), **self.as_dict()}


def compute_binary_metrics(
    predicted_bits: torch.Tensor,
    target_bits: torch.Tensor,
) -> dict[str, float | int]:
    counts = BinaryMetricCounts()
    counts.update(predicted_bits, target_bits)
    return counts.compute()


def compute_metrics_from_logits(
    logits: torch.Tensor,
    target_bits: torch.Tensor,
) -> dict[str, float | int]:
    return compute_binary_metrics(logits_to_bits(logits), target_bits)


def merge_metric_counts(counts: list[BinaryMetricCounts]) -> BinaryMetricCounts:
    merged = BinaryMetricCounts()
    for count in counts:
        merged.bit_errors += count.bit_errors
        merged.block_errors += count.block_errors
        merged.total_bits += count.total_bits
        merged.total_words += count.total_words
    return merged


__all__ = [
    "BinaryMetricCounts",
    "compute_binary_metrics",
    "compute_metrics_from_logits",
    "logits_to_bits",
    "merge_metric_counts",
]
