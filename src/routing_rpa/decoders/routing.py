"""Routing policy base interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
from torch import nn

from routing_rpa.decoders.modes import DecoderMode
from routing_rpa.decoders.selection import ProjectionPlan
from routing_rpa.projections.projection_set import ProjectionSet


class RoutingPolicy(nn.Module, ABC):
    """Base class for projection routing policies."""

    @abstractmethod
    def plan(
        self,
        state: torch.Tensor,
        step: int,
        projections: ProjectionSet,
        mode: DecoderMode,
    ) -> ProjectionPlan:
        raise NotImplementedError


def _weight_dtype(state: torch.Tensor) -> torch.dtype:
    return state.dtype if state.is_floating_point() else torch.float32


def _activation_module(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported router activation: {name!r}")


def _uniform_projection_plan(
    *,
    state: torch.Tensor,
    step: int,
    projections: ProjectionSet,
    mode: DecoderMode,
    routing_policy: str,
) -> ProjectionPlan:
    weights = torch.ones(
        1,
        projections.num_projections,
        device=state.device,
        dtype=_weight_dtype(state),
    )
    return ProjectionPlan(
        candidate_count=projections.num_projections,
        selected_indices=None,
        selection_scope=mode.selection_scope,
        execution_mode=mode.execution_mode,
        aggregation_weights=weights,
        execution_weights=weights,
        aux={
            "routing_policy": routing_policy,
            "frozen_policy": mode.frozen_policy,
            "step": step,
        },
    )


def build_mlp_router(
    input_size: int,
    output_size: int,
    hidden_size: int,
    *,
    use_layer_norm: bool,
    activation: str,
) -> nn.Module:
    """Build a small MLP router network returning projection logits."""
    if input_size <= 0:
        raise ValueError(f"input_size must be positive, got {input_size}")
    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")

    layers: list[nn.Module] = []
    if use_layer_norm:
        layers.append(nn.LayerNorm(input_size))
    layers.extend(
        [
            nn.Linear(input_size, hidden_size),
            _activation_module(activation),
            nn.Linear(hidden_size, output_size),
        ]
    )
    return nn.Sequential(*layers)


class StepwiseRouter(nn.Module):
    """A separate router network for each unfolded decoder step."""

    def __init__(self, routers: Iterable[nn.Module]) -> None:
        super().__init__()
        self.routers = nn.ModuleList(list(routers))
        if len(self.routers) == 0:
            raise ValueError("StepwiseRouter requires at least one router")

    @classmethod
    def from_mlp(
        cls,
        *,
        num_steps: int,
        input_size: int,
        output_size: int,
        hidden_size: int,
        use_layer_norm: bool = False,
        activation: str = "relu",
    ) -> "StepwiseRouter":
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        return cls(
            build_mlp_router(
                input_size=input_size,
                output_size=output_size,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                activation=activation,
            )
            for _ in range(num_steps)
        )

    @property
    def num_steps(self) -> int:
        return len(self.routers)

    def forward(self, state: torch.Tensor, step: int) -> torch.Tensor:
        if state.dim() != 2:
            raise ValueError(f"state must have shape (B, n), got {state.shape}")
        if step < 0 or step >= len(self.routers):
            raise ValueError(f"step must be in [0, {len(self.routers)}), got {step}")
        return self.routers[step](state)


class UniformRouting(RoutingPolicy):
    """Classical uniform-full baseline with no router network."""

    def plan(
        self,
        state: torch.Tensor,
        step: int,
        projections: ProjectionSet,
        mode: DecoderMode,
    ) -> ProjectionPlan:
        if mode.selection_scope != "full":
            raise ValueError("UniformRouting requires selection_scope='full'")
        if mode.execution_mode != "compute_all_mask":
            raise ValueError("UniformRouting requires execution_mode='compute_all_mask'")
        if mode.top_k is not None:
            raise ValueError("UniformRouting uniform_full baseline requires top_k=None")

        weights = torch.ones(
            1,
            projections.num_projections,
            device=state.device,
            dtype=_weight_dtype(state),
        )
        return ProjectionPlan(
            candidate_count=projections.num_projections,
            selected_indices=None,
            selection_scope=mode.selection_scope,
            execution_mode=mode.execution_mode,
            aggregation_weights=weights,
            execution_weights=weights,
            aux={"routing_policy": "uniform_full", "step": step},
        )


class RandomStaticRouting(RoutingPolicy):
    """Seed-controlled fixed random subset baseline."""

    def __init__(self, *, top_k: int | None = None, seed: int = 0) -> None:
        super().__init__()
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive or None, got {top_k}")
        self.top_k = top_k
        self.seed = int(seed)
        self._cached_indices_key: tuple[int, int, torch.device, int] | None = None
        self._cached_indices: torch.Tensor | None = None
        self._cached_projection_key: tuple[int, int, int, torch.device, int] | None = None
        self._cached_selected_projections: ProjectionSet | None = None

    def plan(
        self,
        state: torch.Tensor,
        step: int,
        projections: ProjectionSet,
        mode: DecoderMode,
    ) -> ProjectionPlan:
        if mode.selection_scope != "static":
            raise ValueError("RandomStaticRouting requires selection_scope='static'")
        if mode.execution_mode not in {"compute_selected", "compute_all_mask"}:
            raise ValueError(
                "RandomStaticRouting supports compute_selected and diagnostic compute_all_mask"
            )

        top_k = self._resolve_top_k(mode)
        selected_indices = self._selected_indices(
            projections.num_projections,
            top_k,
            device=state.device,
        )

        if mode.execution_mode == "compute_selected":
            weights = torch.ones(1, top_k, device=state.device, dtype=_weight_dtype(state))
            selected_projections = self._selected_projections(
                projections,
                selected_indices,
                top_k,
            )
            return ProjectionPlan(
                candidate_count=projections.num_projections,
                selected_indices=selected_indices,
                selection_scope=mode.selection_scope,
                execution_mode=mode.execution_mode,
                aggregation_weights=weights,
                execution_weights=weights,
                aux={
                    "routing_policy": "uniform_random_topk",
                    "selected_projections": selected_projections,
                    "step": step,
                },
            )

        aggregation_weights = torch.zeros(
            1,
            projections.num_projections,
            device=state.device,
            dtype=_weight_dtype(state),
        )
        aggregation_weights.scatter_(1, selected_indices.unsqueeze(0), 1.0)
        execution_weights = torch.ones_like(aggregation_weights)
        return ProjectionPlan(
            candidate_count=projections.num_projections,
            selected_indices=selected_indices,
            selection_scope=mode.selection_scope,
            execution_mode=mode.execution_mode,
            aggregation_weights=aggregation_weights,
            execution_weights=execution_weights,
            aux={
                "aggregated_count": top_k,
                "routing_policy": "uniform_random_topk_diagnostic",
                "step": step,
            },
        )

    def _resolve_top_k(self, mode: DecoderMode) -> int:
        if self.top_k is not None and mode.top_k is not None and self.top_k != mode.top_k:
            raise ValueError(
                f"RandomStaticRouting top_k mismatch: constructor={self.top_k}, "
                f"mode={mode.top_k}"
            )
        top_k = mode.top_k if mode.top_k is not None else self.top_k
        if top_k is None:
            raise ValueError("RandomStaticRouting requires top_k in constructor or mode")
        return int(top_k)

    def _selected_indices(
        self,
        candidate_count: int,
        top_k: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if top_k <= 0 or top_k > candidate_count:
            raise ValueError(f"top_k must be in [1, {candidate_count}], got {top_k}")

        target_device = torch.device(device)
        cache_key = (candidate_count, top_k, target_device, self.seed)
        if self._cached_indices_key == cache_key and self._cached_indices is not None:
            return self._cached_indices

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        indices = torch.randperm(candidate_count, generator=generator, dtype=torch.long)
        selected = indices[:top_k].to(device=target_device)
        self._cached_indices_key = cache_key
        self._cached_indices = selected
        return selected

    def _selected_projections(
        self,
        projections: ProjectionSet,
        selected_indices: torch.Tensor,
        top_k: int,
    ) -> ProjectionSet:
        projection_device = projections.coset_indices.device
        cache_key = (
            id(projections),
            projections.num_projections,
            top_k,
            projection_device,
            self.seed,
        )
        cache_matches = (
            self._cached_projection_key == cache_key
            and self._cached_selected_projections is not None
        )
        if cache_matches:
            return self._cached_selected_projections

        selected = selected_indices.to(device=projection_device)
        selected_projections = projections.subset(selected)
        self._cached_projection_key = cache_key
        self._cached_selected_projections = selected_projections
        return selected_projections


class SelectedStaticRouting(RoutingPolicy):
    """Fixed static selected subset for honest compute_selected pruning."""

    def __init__(
        self,
        selected_indices: torch.Tensor,
        *,
        projection_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(selected_indices, torch.Tensor):
            raise TypeError("selected_indices must be a torch.Tensor")
        if selected_indices.dim() != 1:
            raise ValueError(
                f"selected_indices must have shape (K,), got {selected_indices.shape}"
            )
        if selected_indices.numel() == 0:
            raise ValueError("selected_indices must select at least one projection")
        if selected_indices.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError(f"selected_indices must use an integer dtype, got {selected_indices.dtype}")
        if torch.unique(selected_indices.to(dtype=torch.long)).numel() != selected_indices.numel():
            raise ValueError("selected_indices must not contain duplicate projections")

        indices = selected_indices.to(dtype=torch.long).clone()
        self.register_buffer("selected_indices", indices)
        self._min_selected_index = int(indices.min().item())
        self._max_selected_index = int(indices.max().item())
        self._cached_projection_source_id: int | None = None
        self._cached_projection_device: torch.device | None = None
        self._cached_selected_projections: ProjectionSet | None = None

        if projection_weights is None:
            self.register_buffer("fixed_projection_weights", torch.empty(0))
        else:
            if not isinstance(projection_weights, torch.Tensor):
                raise TypeError("projection_weights must be a torch.Tensor or None")
            if projection_weights.dim() != 1:
                raise ValueError(
                    f"projection_weights must have shape (K,), got {projection_weights.shape}"
                )
            if projection_weights.numel() != indices.numel():
                raise ValueError(
                    "projection_weights length must match selected_indices length: "
                    f"{projection_weights.numel()} != {indices.numel()}"
                )
            self.register_buffer("fixed_projection_weights", projection_weights.clone())

    def plan(
        self,
        state: torch.Tensor,
        step: int,
        projections: ProjectionSet,
        mode: DecoderMode,
    ) -> ProjectionPlan:
        if mode.selection_scope != "static":
            raise ValueError("SelectedStaticRouting requires selection_scope='static'")
        if mode.execution_mode != "compute_selected":
            raise ValueError("SelectedStaticRouting requires execution_mode='compute_selected'")

        indices = self.selected_indices.to(device=state.device)
        if self._min_selected_index < 0 or self._max_selected_index >= projections.num_projections:
            raise ValueError(
                f"selected_indices must be in [0, {projections.num_projections})"
            )
        if mode.top_k is not None and mode.top_k != indices.numel():
            raise ValueError(
                f"SelectedStaticRouting top_k mismatch: mode={mode.top_k}, "
                f"selected_count={indices.numel()}"
            )

        if self.fixed_projection_weights.numel() == 0:
            weights = torch.ones(
                1,
                indices.numel(),
                device=state.device,
                dtype=_weight_dtype(state),
            )
        else:
            weights = self.fixed_projection_weights.to(
                device=state.device,
                dtype=_weight_dtype(state),
            ).unsqueeze(0)
        selected_projections = self._selected_projections(projections)

        return ProjectionPlan(
            candidate_count=projections.num_projections,
            selected_indices=indices,
            selection_scope=mode.selection_scope,
            execution_mode=mode.execution_mode,
            aggregation_weights=weights,
            execution_weights=weights,
            aux={
                "routing_policy": "selected_static_topk",
                "selected_projections": selected_projections,
                "step": step,
            },
        )

    def _selected_projections(self, projections: ProjectionSet) -> ProjectionSet:
        cache_matches = (
            self._cached_projection_source_id == id(projections)
            and self._cached_projection_device == projections.coset_indices.device
            and self._cached_selected_projections is not None
        )
        if cache_matches:
            return self._cached_selected_projections

        selected = self.selected_indices.to(device=projections.coset_indices.device)
        selected_projections = projections.subset(selected)
        self._cached_projection_source_id = id(projections)
        self._cached_projection_device = projections.coset_indices.device
        self._cached_selected_projections = selected_projections
        return selected_projections


class InputDependentRouterRouting(RoutingPolicy):
    """Input-dependent projection weighting with diagnostic compute-all top-k."""

    def __init__(
        self,
        router: StepwiseRouter,
        *,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive or None, got {top_k}")
        self.router = router
        self.top_k = top_k

    def plan(
        self,
        state: torch.Tensor,
        step: int,
        projections: ProjectionSet,
        mode: DecoderMode,
    ) -> ProjectionPlan:
        if mode.selection_scope != "full":
            raise ValueError("InputDependentRouterRouting requires selection_scope='full'")
        if mode.execution_mode != "compute_all_mask":
            raise ValueError(
                "InputDependentRouterRouting stage-9 path only supports "
                "execution_mode='compute_all_mask'"
            )
        if mode.uses_uniform_frozen_weights(step):
            return _uniform_projection_plan(
                state=state,
                step=step,
                projections=projections,
                mode=mode,
                routing_policy="uniform_frozen",
            )

        logits = self.router(state, step)
        if logits.shape != (state.shape[0], projections.num_projections):
            raise ValueError(
                "router logits must have shape "
                f"({state.shape[0]}, {projections.num_projections}), got {logits.shape}"
            )

        sigmoid_weights = torch.sigmoid(logits)
        top_k = self._resolve_top_k(mode)
        selected_indices: torch.Tensor | None = None
        aux: dict[str, object] = {
            "routing_policy": "input_dependent_router",
            "router_logits": logits,
            "router_entropy_inputs": logits,
            "step": step,
        }

        if top_k is None:
            aggregation_weights = sigmoid_weights
        else:
            if top_k > projections.num_projections:
                raise ValueError(
                    f"top_k must be in [1, {projections.num_projections}], got {top_k}"
                )
            selected_indices = torch.topk(logits, top_k, dim=1).indices
            mask = torch.zeros_like(logits)
            mask.scatter_(1, selected_indices, 1.0)
            aggregation_weights = sigmoid_weights * mask
            aux["aggregated_count"] = top_k
            aux["routing_policy"] = "full_masked_topk"

        execution_weights = torch.ones(
            state.shape[0],
            projections.num_projections,
            device=state.device,
            dtype=_weight_dtype(state),
        )
        return ProjectionPlan(
            candidate_count=projections.num_projections,
            selected_indices=selected_indices,
            selection_scope=mode.selection_scope,
            execution_mode=mode.execution_mode,
            aggregation_weights=aggregation_weights,
            execution_weights=execution_weights,
            aux=aux,
        )

    def _resolve_top_k(self, mode: DecoderMode) -> int | None:
        if self.top_k is not None and mode.top_k is not None and self.top_k != mode.top_k:
            raise ValueError(
                f"InputDependentRouterRouting top_k mismatch: constructor={self.top_k}, "
                f"mode={mode.top_k}"
            )
        return mode.top_k if mode.top_k is not None else self.top_k


__all__ = [
    "InputDependentRouterRouting",
    "RandomStaticRouting",
    "RoutingPolicy",
    "SelectedStaticRouting",
    "StepwiseRouter",
    "UniformRouting",
    "build_mlp_router",
]
