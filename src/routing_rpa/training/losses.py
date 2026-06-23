"""Config-driven loss composition for decoder training ablations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from routing_rpa.decoders.outputs import DecoderOutput


_VALID_COMPONENTS = {
    "bce_logits",
    "soft_ber",
    "legacy_elementary_ce",
    "router_entropy",
}


@dataclass(frozen=True)
class LossComponentConfig:
    name: str
    weight: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LossConfig:
    components: Sequence[LossComponentConfig | Mapping[str, Any]]


def _normalize_component(
    component: LossComponentConfig | Mapping[str, Any],
) -> LossComponentConfig:
    if isinstance(component, LossComponentConfig):
        normalized = component
    elif isinstance(component, Mapping):
        params = {
            key: value
            for key, value in component.items()
            if key not in {"name", "weight", "params"}
        }
        params.update(dict(component.get("params", {})))
        normalized = LossComponentConfig(
            name=str(component["name"]),
            weight=float(component["weight"]),
            params=params,
        )
    else:
        raise TypeError(f"Unsupported loss component config type: {type(component)!r}")

    if normalized.name not in _VALID_COMPONENTS:
        raise ValueError(f"Unsupported loss component: {normalized.name!r}")
    if normalized.weight < 0:
        raise ValueError(
            f"Loss component weight must be non-negative, got {normalized.weight}"
        )
    return normalized


def _validate_logits_and_target(output: DecoderOutput, target_codeword: torch.Tensor) -> None:
    if output.logits.shape != target_codeword.shape:
        raise ValueError(
            f"logits and target_codeword must have the same shape, "
            f"got {output.logits.shape} and {target_codeword.shape}"
        )
    if output.logits.dim() != 2:
        raise ValueError(f"losses expect logits shape (B, n), got {output.logits.shape}")


def _soft_ber_loss(
    logits: torch.Tensor,
    target_codeword: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    y = target_codeword.float() * 2.0 - 1.0
    return torch.sigmoid(-beta * y * logits).mean()


def _legacy_elementary_ce_loss(
    logits: torch.Tensor,
    target_codeword: torch.Tensor,
) -> torch.Tensor:
    losses = [
        F.binary_cross_entropy_with_logits(
            logits[:, index],
            target_codeword[:, index].float(),
        )
        for index in range(logits.shape[1])
    ]
    return torch.stack(losses).sum()


def _collect_router_entropy_inputs(aux: dict[str, Any]) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    direct = aux.get("router_entropy_inputs")
    if isinstance(direct, torch.Tensor):
        tensors.append(direct)

    for step in aux.get("steps", []):
        if not isinstance(step, Mapping):
            continue
        plan_aux = step.get("plan_aux", {})
        if not isinstance(plan_aux, Mapping):
            continue
        entropy_inputs = plan_aux.get("router_entropy_inputs")
        if isinstance(entropy_inputs, torch.Tensor):
            tensors.append(entropy_inputs)
    return tensors


def _router_entropy_loss(output: DecoderOutput, *, eps: float) -> torch.Tensor:
    entropy_inputs = _collect_router_entropy_inputs(output.aux)
    if not entropy_inputs:
        raise ValueError("router_entropy loss requires router_entropy_inputs in DecoderOutput.aux")

    entropy_terms = []
    for logits in entropy_inputs:
        probs = torch.softmax(logits, dim=-1)
        entropy_terms.append(-(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1).mean())
    return torch.stack(entropy_terms).mean()


class LossComposer:
    """Combine supervised losses and regularizers from an explicit config."""

    def __init__(self, config: LossConfig) -> None:
        self.components = [_normalize_component(component) for component in config.components]
        if not self.components:
            raise ValueError("LossConfig must contain at least one component")

    def __call__(
        self,
        output: DecoderOutput,
        target_codeword: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        _validate_logits_and_target(output, target_codeword)

        total = output.logits.sum() * 0.0
        scalars: dict[str, float] = {}
        for component in self.components:
            if component.weight == 0.0:
                scalars[component.name] = 0.0
                scalars[f"{component.name}_weighted"] = 0.0
                continue

            value = self._component_value(component, output, target_codeword)
            weighted = value * component.weight
            total = total + weighted
            scalars[component.name] = float(value.detach().item())
            scalars[f"{component.name}_weighted"] = float(weighted.detach().item())

        scalars["total"] = float(total.detach().item())
        return total, scalars

    @staticmethod
    def _component_value(
        component: LossComponentConfig,
        output: DecoderOutput,
        target_codeword: torch.Tensor,
    ) -> torch.Tensor:
        if component.name == "bce_logits":
            return F.binary_cross_entropy_with_logits(
                output.logits,
                target_codeword.float(),
            )
        if component.name == "soft_ber":
            beta = float(component.params.get("beta", 5.0))
            return _soft_ber_loss(output.logits, target_codeword, beta=beta)
        if component.name == "legacy_elementary_ce":
            return _legacy_elementary_ce_loss(output.logits, target_codeword)
        if component.name == "router_entropy":
            eps = float(component.params.get("eps", 1e-12))
            return _router_entropy_loss(output, eps=eps)
        raise AssertionError(f"Unhandled loss component: {component.name}")


__all__ = [
    "LossComponentConfig",
    "LossComposer",
    "LossConfig",
]
