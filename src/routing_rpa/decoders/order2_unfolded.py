"""Clean order-2 unfolded RPA decoder backend."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from routing_rpa.codes.linear import LinearCode
from routing_rpa.decoders.bottom_decoders import BottomDecoder
from routing_rpa.decoders.execution import ProjectionExecutor
from routing_rpa.decoders.kernels_order2 import aggregate_1d, project_1d
from routing_rpa.decoders.modes import DecoderMode
from routing_rpa.decoders.outputs import DecoderOutput
from routing_rpa.decoders.routing import RoutingPolicy
from routing_rpa.projections.projection_set import ProjectionSet


class ChannelLLR(Protocol):
    """Callable that converts channel outputs to length-n LLR tensors."""

    def __call__(self, received: Tensor, *args: Any) -> Tensor:
        ...


def resolve_unfolded_steps(num_unfolded_steps: int, mode: DecoderMode) -> int:
    """Resolve decoder depth through the explicit DecoderMode contract."""
    return mode.resolve_num_steps(num_unfolded_steps)


class Order2UnfoldedRPADecoder(nn.Module):
    """Order-2 unfolded RPA decoder with mode-driven forward behavior."""

    def __init__(
        self,
        code: LinearCode,
        projections: ProjectionSet,
        bottom_decoder: BottomDecoder,
        routing_policy: RoutingPolicy,
        num_unfolded_steps: int,
        channel_llr: ChannelLLR,
    ) -> None:
        super().__init__()
        if code.n != projections.n:
            raise ValueError(
                f"code.n must match projections.n: {code.n} != {projections.n}"
            )
        if num_unfolded_steps <= 0:
            raise ValueError(
                f"num_unfolded_steps must be positive, got {num_unfolded_steps}"
            )

        self.code = code
        self.projections = projections
        self.bottom_decoder = bottom_decoder
        self.routing_policy = routing_policy
        self.num_unfolded_steps = int(num_unfolded_steps)
        self.channel_llr = channel_llr
        self.projection_executor = ProjectionExecutor()

    def _apply(self, fn):
        super()._apply(fn)
        self.code.generator_matrix = fn(self.code.generator_matrix)
        self.projections = ProjectionSet._from_trusted_parts(
            m=self.projections.m,
            n=self.projections.n,
            subspace_dim=self.projections.subspace_dim,
            directions=fn(self.projections.directions),
            coset_indices=fn(self.projections.coset_indices),
            metadata=dict(self.projections.metadata),
            flat_ids1=fn(self.projections.flat_ids1),
            flat_ids2=fn(self.projections.flat_ids2),
        )
        return self

    def forward(
        self,
        channel_output: Tensor,
        mode: DecoderMode,
    ) -> DecoderOutput:
        if channel_output.dim() != 2:
            raise ValueError(
                f"channel_output must have shape (B, n), got {channel_output.shape}"
            )
        if channel_output.shape[1] != self.code.n:
            raise ValueError(
                f"channel_output length must match code.n={self.code.n}, "
                f"got {channel_output.shape[1]}"
            )

        state = self._convert_to_llr(channel_output, mode)
        if state.shape != channel_output.shape:
            raise ValueError(
                "channel_llr must return a tensor with the same shape as channel_output: "
                f"{state.shape} != {channel_output.shape}"
            )

        steps = resolve_unfolded_steps(self.num_unfolded_steps, mode)
        aux_steps: list[dict[str, object]] = []
        per_step_stats: list[dict[str, object]] = []

        latest_stats: dict[str, object] = {}
        for step in range(steps):
            run_no_grad = mode.runs_no_grad(step)
            step_context = torch.no_grad() if run_no_grad else nullcontext()
            with step_context:
                plan = self.routing_policy.plan(
                    state=state,
                    step=step,
                    projections=self.projections,
                    mode=mode,
                )
                execution = self.projection_executor.resolve(self.projections, plan)

                projected = project_1d(state, execution.projections)
                decoded = self.bottom_decoder(projected, execution.projections, step=step)
                state = aggregate_1d(
                    received_llr=state,
                    decoded_projected=decoded,
                    projection_weights=execution.aggregation_weights,
                    projections=execution.projections,
                )

            latest_stats = dict(execution.stats)
            latest_stats["frozen_policy"] = mode.frozen_policy
            latest_stats["ran_no_grad"] = run_no_grad
            per_step_stats.append({"step": step, **latest_stats})
            step_aux = self._build_step_aux(
                step=step,
                plan_aux=plan.aux,
                selected_indices=plan.selected_indices,
                aggregation_weights=execution.aggregation_weights,
                execution_weights=execution.execution_weights,
                collect_debug=mode.collect_debug,
            )
            if len(step_aux) > 1:
                aux_steps.append(step_aux)

        logits = -state
        stats = {
            **latest_stats,
            "num_unfolded_steps_configured": self.num_unfolded_steps,
            "num_unfolded_steps_executed": steps,
            "per_step_stats": per_step_stats,
        }
        aux = {"steps": aux_steps}
        return DecoderOutput(logits=logits, aux=aux, stats=stats)

    def _convert_to_llr(self, channel_output: Tensor, mode: DecoderMode) -> Tensor:
        context = mode.channel_context
        if "snr" in context:
            return self.channel_llr(channel_output, float(context["snr"]), self.code)
        if context:
            return self.channel_llr(channel_output, context, self.code)
        return self.channel_llr(channel_output, self.code)

    @staticmethod
    def _build_step_aux(
        *,
        step: int,
        plan_aux: Mapping[str, object],
        selected_indices: Tensor | None,
        aggregation_weights: Tensor,
        execution_weights: Tensor,
        collect_debug: bool,
    ) -> dict[str, object]:
        step_aux: dict[str, object] = {"step": step}

        loss_plan_aux: dict[str, object] = {}
        if "router_entropy_inputs" in plan_aux:
            loss_plan_aux["router_entropy_inputs"] = plan_aux["router_entropy_inputs"]
        if loss_plan_aux:
            step_aux["plan_aux"] = loss_plan_aux

        if collect_debug:
            debug_plan_aux = dict(plan_aux)
            if loss_plan_aux:
                debug_plan_aux.update(loss_plan_aux)
            step_aux.update(
                {
                    "plan_aux": debug_plan_aux,
                    "selected_indices": selected_indices,
                    "aggregation_weights": aggregation_weights,
                    "execution_weights": execution_weights,
                }
            )
        return step_aux


__all__ = [
    "ChannelLLR",
    "Order2UnfoldedRPADecoder",
    "resolve_unfolded_steps",
]
