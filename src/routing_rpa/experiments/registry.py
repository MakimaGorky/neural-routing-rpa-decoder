"""Small named registries for experiment assembly."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from routing_rpa.channels.awgn import AWGNChannel, AWGNLLR
from routing_rpa.channels.fading import AWGNFadingChannel
from routing_rpa.channels.reliability_profiles import CoordinateReliabilityAWGN
from routing_rpa.codes.reed_muller import RMCode
from routing_rpa.decoders.bottom_decoders import BottomDecoder, HadamardOrder1Decoder
from routing_rpa.decoders.order2_unfolded import Order2UnfoldedRPADecoder
from routing_rpa.decoders.routing import (
    InputDependentRouterRouting,
    RandomStaticRouting,
    RoutingPolicy,
    SelectedStaticRouting,
    StepwiseRouter,
    UniformRouting,
)
from routing_rpa.projections.projection_set import ProjectionSet
from routing_rpa.training.checkpoints import build_checkpoint_policy

from routing_rpa.experiments.config import (
    ChannelConfig,
    CodeConfig,
    DecoderConfig,
    RoutingConfig,
)


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"float32", "torch.float32", "fp32"}:
        return torch.float32
    if normalized in {"float64", "torch.float64", "double", "fp64"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name!r}")


def build_code(
    config: CodeConfig,
    *,
    artifact_path: Path,
    device: torch.device,
) -> RMCode:
    family = config.family.lower()
    if family == "rm":
        return RMCode.from_artifact(
            artifact_path,
            m=config.m,
            r=config.r,
            dtype=dtype_from_name(config.dtype),
            device=device,
        )
    if family == "rm_subcode":
        raise NotImplementedError("rm_subcode build support is reserved for the subcode stage")
    raise ValueError(f"Unsupported code family: {config.family!r}")


def build_channel(config: ChannelConfig) -> AWGNChannel:
    name = config.name.lower()
    if name == "awgn":
        return AWGNChannel()
    if name == "awgn_fading":
        return AWGNFadingChannel(**config.kwargs)
    if name == "coordinate_reliability_awgn":
        return CoordinateReliabilityAWGN(**config.kwargs)
    raise ValueError(f"Unsupported channel: {config.name!r}")


def build_bottom_decoder(
    config: DecoderConfig,
    projections: ProjectionSet,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> BottomDecoder:
    if config.bottom_decoder != "hadamard_order1":
        if config.bottom_decoder == "soft_map_subcode":
            raise NotImplementedError(
                "soft_map_subcode bottom decoder is reserved for the subcode stage"
            )
        raise ValueError(f"Unsupported bottom decoder: {config.bottom_decoder!r}")
    return HadamardOrder1Decoder(length=projections.n // 2, dtype=dtype).to(device)


def build_routing_policy(
    config: RoutingConfig,
    *,
    code_n: int,
    projection_count: int,
    num_unfolded_steps: int,
) -> RoutingPolicy:
    policy = config.policy.lower()
    if policy == "uniform_full":
        return UniformRouting()
    if policy == "random_static":
        return RandomStaticRouting(top_k=config.top_k, seed=config.random_seed)
    if policy == "selected_static_topk":
        selected = _selected_indices_tensor(config, projection_count)
        weights = (
            torch.tensor(config.projection_weights, dtype=torch.float32)
            if config.projection_weights is not None
            else None
        )
        return SelectedStaticRouting(selected, projection_weights=weights)
    if policy in {"input_dependent_router", "full_masked_topk"}:
        router_name = config.router or "mlp"
        if router_name != "mlp":
            raise ValueError(f"Unsupported router network: {router_name!r}")
        router = StepwiseRouter.from_mlp(
            num_steps=num_unfolded_steps,
            input_size=code_n,
            output_size=projection_count,
            hidden_size=config.hidden_size,
            use_layer_norm=config.use_layer_norm,
            activation=config.activation,
        )
        return InputDependentRouterRouting(router, top_k=config.top_k)
    raise ValueError(f"Unsupported routing policy: {config.policy!r}")


def build_decoder_backend(
    config: DecoderConfig,
    *,
    code: RMCode,
    projections: ProjectionSet,
    bottom_decoder: BottomDecoder,
    routing_policy: RoutingPolicy,
) -> Order2UnfoldedRPADecoder:
    if config.backend != "order2_unfolded":
        raise ValueError(f"Unsupported decoder backend: {config.backend!r}")
    return Order2UnfoldedRPADecoder(
        code=code,
        projections=projections,
        bottom_decoder=bottom_decoder,
        routing_policy=routing_policy,
        num_unfolded_steps=config.num_unfolded_steps,
        channel_llr=AWGNLLR(),
    )


def build_optimizer_factory(name: str, *, lr: float) -> Callable[[Any], torch.optim.Optimizer]:
    normalized = name.lower()
    if normalized == "adam":
        return lambda params: torch.optim.Adam(params, lr=lr)
    if normalized == "sgd":
        return lambda params: torch.optim.SGD(params, lr=lr)
    raise ValueError(f"Unsupported optimizer: {name!r}")


def build_checkpoint_policy_by_name(name: str):
    return build_checkpoint_policy(name)


def _selected_indices_tensor(config: RoutingConfig, projection_count: int) -> torch.Tensor:
    if config.selected_indices is None:
        if config.top_k is None:
            raise ValueError("selected_static_topk requires selected_indices or top_k")
        selected_indices = tuple(range(config.top_k))
    else:
        selected_indices = config.selected_indices

    selected = torch.tensor(selected_indices, dtype=torch.long)
    if selected.numel() == 0:
        raise ValueError("selected_static_topk must select at least one projection")
    if int(selected.min().item()) < 0 or int(selected.max().item()) >= projection_count:
        raise ValueError(
            f"selected_static_topk indices must be in [0, {projection_count})"
        )
    if config.top_k is not None and selected.numel() != config.top_k:
        raise ValueError(
            f"routing.top_k={config.top_k} does not match selected_indices length "
            f"{selected.numel()}"
        )
    return selected


__all__ = [
    "build_bottom_decoder",
    "build_channel",
    "build_checkpoint_policy_by_name",
    "build_code",
    "build_decoder_backend",
    "build_optimizer_factory",
    "build_routing_policy",
    "dtype_from_name",
]
