"""Explicit training strategies for decoder orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from torch import nn

from routing_rpa.decoders.modes import (
    DecoderMode,
    ExecutionMode,
    ForwardDepthPolicy,
    FrozenPolicy,
    SelectionScope,
)


class TrainingStrategy(Protocol):
    def layers_to_train(self, model: nn.Module) -> list[int]:
        ...

    def mode_for_step(self, layer_id: int | None, phase: str) -> DecoderMode:
        ...

    def trainable_parameters(
        self,
        model: nn.Module,
        layer_id: int | None,
    ) -> Iterable[nn.Parameter]:
        ...


@dataclass(frozen=True)
class LayerwiseTraining:
    """Train one routing/global-weight layer at a time."""

    selection_scope: SelectionScope = "full"
    execution_mode: ExecutionMode = "compute_all_mask"
    top_k: int | None = None
    forward_depth_policy: ForwardDepthPolicy = "local_depth"
    frozen_policy: FrozenPolicy = "frozen_weights"
    train_layers: Sequence[int] | None = None
    collect_debug: bool = False

    def __post_init__(self) -> None:
        if self.train_layers is not None:
            object.__setattr__(self, "train_layers", tuple(int(layer) for layer in self.train_layers))
        target_layer = 0 if self.forward_depth_policy == "local_depth" else None
        DecoderMode(
            selection_scope=self.selection_scope,
            execution_mode=self.execution_mode,
            top_k=self.top_k,
            forward_depth_policy=self.forward_depth_policy,
            frozen_policy=self.frozen_policy,
            target_layer=target_layer,
            collect_debug=self.collect_debug,
        )

    def layers_to_train(self, model: nn.Module) -> list[int]:
        num_layers = _num_trainable_layers(model)
        if self.train_layers is None:
            return list(range(num_layers))

        layers = list(self.train_layers)
        seen: set[int] = set()
        for layer_id in layers:
            if layer_id in seen:
                raise ValueError(f"train_layers contains duplicate layer {layer_id}")
            seen.add(layer_id)
            if layer_id < 0 or layer_id >= num_layers:
                raise ValueError(
                    f"train layer {layer_id} is outside available layers [0, {num_layers})"
                )
        return layers

    def mode_for_step(self, layer_id: int | None, phase: str) -> DecoderMode:
        if not phase:
            raise ValueError("phase must be non-empty")

        if layer_id is None:
            forward_depth_policy: ForwardDepthPolicy = "full_cascade"
            target_layer = None
        else:
            target_layer = int(layer_id)
            forward_depth_policy = self.forward_depth_policy
            if self.frozen_policy == "skip_future":
                forward_depth_policy = "local_depth"

        return DecoderMode(
            selection_scope=self.selection_scope,
            execution_mode=self.execution_mode,
            top_k=self.top_k,
            forward_depth_policy=forward_depth_policy,
            frozen_policy=self.frozen_policy,
            target_layer=target_layer,
            collect_debug=self.collect_debug,
        )

    def trainable_parameters(
        self,
        model: nn.Module,
        layer_id: int | None,
    ) -> Iterable[nn.Parameter]:
        if layer_id is None:
            raise ValueError("LayerwiseTraining requires an explicit layer_id")

        _validate_layer_id(model, layer_id)
        for parameter in _routing_parameters(model):
            parameter.requires_grad_(False)

        active_parameters = list(_parameters_for_layer(model, layer_id))
        if not active_parameters:
            raise ValueError(f"layer {layer_id} has no trainable parameters")
        for parameter in active_parameters:
            parameter.requires_grad_(True)
        return active_parameters


def _routing_policy(model: nn.Module) -> object:
    routing_policy = getattr(model, "routing_policy", None)
    if routing_policy is None:
        raise AttributeError("model must expose routing_policy for layerwise training")
    return routing_policy


def _router_layers(model: nn.Module) -> list[nn.Module]:
    routing_policy = _routing_policy(model)
    if hasattr(routing_policy, "layer_modules"):
        return list(routing_policy.layer_modules())

    router = getattr(routing_policy, "router", None)
    routers = getattr(router, "routers", None)
    if routers is None:
        routers = getattr(routing_policy, "routers", None)
    if routers is None:
        return []
    return list(routers)


def _num_trainable_layers(model: nn.Module) -> int:
    layers = _router_layers(model)
    if layers:
        return len(layers)

    num_steps = getattr(model, "num_unfolded_steps", None)
    if isinstance(num_steps, int) and num_steps > 0:
        return num_steps
    raise ValueError("could not infer trainable layer count from model")


def _validate_layer_id(model: nn.Module, layer_id: int) -> None:
    num_layers = _num_trainable_layers(model)
    if layer_id < 0 or layer_id >= num_layers:
        raise ValueError(f"layer_id must be in [0, {num_layers}), got {layer_id}")


def _parameters_for_layer(model: nn.Module, layer_id: int) -> Iterable[nn.Parameter]:
    routing_policy = _routing_policy(model)
    parameters_for_layer = getattr(routing_policy, "parameters_for_layer", None)
    if callable(parameters_for_layer):
        return parameters_for_layer(layer_id)

    layers = _router_layers(model)
    if not layers:
        raise ValueError("routing policy does not expose per-layer parameters")
    return layers[layer_id].parameters()


def _routing_parameters(model: nn.Module) -> list[nn.Parameter]:
    routing_policy = _routing_policy(model)
    if isinstance(routing_policy, nn.Module):
        return list(routing_policy.parameters())

    parameters = []
    for layer in _router_layers(model):
        parameters.extend(layer.parameters())
    return parameters


__all__ = [
    "LayerwiseTraining",
    "TrainingStrategy",
]
