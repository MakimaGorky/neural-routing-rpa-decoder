"""Explicit checkpoint selection policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


_MONITORED_KEYS = ("bler", "ber")


@dataclass(frozen=True)
class CheckpointDecision:
    should_save: bool
    policy_name: str
    reason: str
    metadata: dict[str, object]


def _extract_monitored_metrics(metrics: Mapping[str, float | int]) -> dict[str, float]:
    missing = [key for key in _MONITORED_KEYS if key not in metrics]
    if missing:
        raise ValueError(f"checkpoint metrics missing required keys: {missing}")
    return {key: float(metrics[key]) for key in _MONITORED_KEYS}


def build_checkpoint_metadata(
    *,
    policy_name: str,
    metrics: Mapping[str, float | int],
    epoch: int | None = None,
    step: int | None = None,
    layer_id: int | None = None,
    config_snapshot_path: str | None = None,
    config_snapshot_hash: str | None = None,
    previous_best: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Build structured metadata for a checkpoint decision."""
    monitored_metrics = _extract_monitored_metrics(metrics)
    metadata: dict[str, object] = {
        "policy_name": policy_name,
        "monitored_metrics": monitored_metrics,
        "epoch": epoch,
        "step": step,
        "layer_id": layer_id,
        "config_snapshot_path": config_snapshot_path,
        "config_snapshot_hash": config_snapshot_hash,
    }
    if previous_best is not None:
        metadata["previous_best"] = dict(previous_best)
    return metadata


class CheckpointPolicy:
    """Base stateful policy that tracks the best saved metrics."""

    name: str

    def __init__(self) -> None:
        self.best_metrics: dict[str, float] | None = None

    def reset(self) -> None:
        self.best_metrics = None

    def decide(
        self,
        metrics: Mapping[str, float | int],
        *,
        epoch: int | None = None,
        step: int | None = None,
        layer_id: int | None = None,
        config_snapshot_path: str | None = None,
        config_snapshot_hash: str | None = None,
    ) -> CheckpointDecision:
        current = _extract_monitored_metrics(metrics)
        previous_best = None if self.best_metrics is None else dict(self.best_metrics)
        should_save, reason = self._should_save(current)
        metadata = build_checkpoint_metadata(
            policy_name=self.name,
            metrics=metrics,
            epoch=epoch,
            step=step,
            layer_id=layer_id,
            config_snapshot_path=config_snapshot_path,
            config_snapshot_hash=config_snapshot_hash,
            previous_best=previous_best,
        )

        if should_save:
            self.best_metrics = current

        return CheckpointDecision(
            should_save=should_save,
            policy_name=self.name,
            reason=reason,
            metadata=metadata,
        )

    def _should_save(self, current: Mapping[str, float]) -> tuple[bool, str]:
        raise NotImplementedError


class BothImproveCheckpointPolicy(CheckpointPolicy):
    """Save only when BLER and BER both improve."""

    name = "both_improve"

    def _should_save(self, current: Mapping[str, float]) -> tuple[bool, str]:
        if self.best_metrics is None:
            return True, "first_checkpoint"
        if current["bler"] < self.best_metrics["bler"] and current["ber"] < self.best_metrics["ber"]:
            return True, "bler_and_ber_improved"
        return False, "requires_both_bler_and_ber_improvement"


class BlerPrimaryWithBERTieBreakerPolicy(CheckpointPolicy):
    """Save on lower BLER; use BER only when BLER ties."""

    name = "bler_primary_with_ber_tiebreaker"

    def _should_save(self, current: Mapping[str, float]) -> tuple[bool, str]:
        if self.best_metrics is None:
            return True, "first_checkpoint"
        if current["bler"] < self.best_metrics["bler"]:
            return True, "bler_improved"
        if current["bler"] == self.best_metrics["bler"] and current["ber"] < self.best_metrics["ber"]:
            return True, "bler_tie_ber_improved"
        return False, "bler_not_improved"


def build_checkpoint_policy(name: str) -> CheckpointPolicy:
    if name == BothImproveCheckpointPolicy.name:
        return BothImproveCheckpointPolicy()
    if name == BlerPrimaryWithBERTieBreakerPolicy.name:
        return BlerPrimaryWithBERTieBreakerPolicy()
    raise ValueError(f"Unsupported checkpoint policy: {name!r}")


__all__ = [
    "BlerPrimaryWithBERTieBreakerPolicy",
    "BothImproveCheckpointPolicy",
    "CheckpointDecision",
    "CheckpointPolicy",
    "build_checkpoint_metadata",
    "build_checkpoint_policy",
]
