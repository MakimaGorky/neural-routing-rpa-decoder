"""Decoder backends, modes, routing, and execution policies."""

from routing_rpa.decoders.bottom_decoders import BottomDecoder, HadamardOrder1Decoder
from routing_rpa.decoders.execution import ProjectionExecution, ProjectionExecutor
from routing_rpa.decoders.kernels_order2 import (
    aggregate_1d,
    convert_to_llr,
    hadamard_decode_order1,
    project_1d,
)
from routing_rpa.decoders.modes import DecoderMode
from routing_rpa.decoders.order2_unfolded import (
    ChannelLLR,
    Order2UnfoldedRPADecoder,
    resolve_unfolded_steps,
)
from routing_rpa.decoders.outputs import DecoderOutput
from routing_rpa.decoders.routing import (
    InputDependentRouterRouting,
    RandomStaticRouting,
    RoutingPolicy,
    SelectedStaticRouting,
    StepwiseRouter,
    UniformRouting,
    build_mlp_router,
)
from routing_rpa.decoders.selection import ProjectionPlan

__all__ = [
    "BottomDecoder",
    "ChannelLLR",
    "DecoderMode",
    "DecoderOutput",
    "HadamardOrder1Decoder",
    "InputDependentRouterRouting",
    "Order2UnfoldedRPADecoder",
    "ProjectionExecution",
    "ProjectionExecutor",
    "ProjectionPlan",
    "RandomStaticRouting",
    "RoutingPolicy",
    "SelectedStaticRouting",
    "StepwiseRouter",
    "UniformRouting",
    "aggregate_1d",
    "convert_to_llr",
    "hadamard_decode_order1",
    "project_1d",
    "resolve_unfolded_steps",
    "build_mlp_router",
]
