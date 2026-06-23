import torch
from torch import nn

from routing_rpa.codes.linear import LinearCode
from routing_rpa.decoders.bottom_decoders import HadamardOrder1Decoder
from routing_rpa.decoders.execution import ProjectionExecutor
from routing_rpa.decoders.modes import DecoderMode
from routing_rpa.decoders.order2_unfolded import Order2UnfoldedRPADecoder
from routing_rpa.decoders.routing import (
    InputDependentRouterRouting,
    StepwiseRouter,
    build_mlp_router,
)
from routing_rpa.projections.projection_set import ProjectionSet


class IdentityLLR:
    def __call__(self, received: torch.Tensor, code: LinearCode) -> torch.Tensor:
        return received


class FailingRouter(nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raise AssertionError("frozen uniform step must not call router")


def make_projection_set() -> ProjectionSet:
    return ProjectionSet.from_coset_indices(
        m=2,
        n=4,
        subspace_dim=1,
        directions=torch.tensor([1, 2, 3], dtype=torch.long),
        coset_indices=torch.tensor(
            [
                [[0, 1], [2, 3]],
                [[0, 2], [1, 3]],
                [[0, 3], [1, 2]],
            ],
            dtype=torch.long,
        ),
        metadata={"name": "toy", "num_projections": 3},
    )


def router_mode(*, top_k: int | None = None) -> DecoderMode:
    return DecoderMode(
        selection_scope="full",
        execution_mode="compute_all_mask",
        top_k=top_k,
        forward_depth_policy="full_cascade",
        frozen_policy="frozen_weights",
    )


def uniform_frozen_mode(*, target_layer: int = 1, top_k: int | None = 2) -> DecoderMode:
    return DecoderMode(
        selection_scope="full",
        execution_mode="compute_all_mask",
        top_k=top_k,
        forward_depth_policy="local_depth",
        frozen_policy="uniform_frozen",
        target_layer=target_layer,
    )


def make_stepwise_router(*, num_steps: int = 1) -> StepwiseRouter:
    return StepwiseRouter.from_mlp(
        num_steps=num_steps,
        input_size=4,
        output_size=3,
        hidden_size=5,
        use_layer_norm=False,
        activation="relu",
    )


def test_build_mlp_router_logits_shape():
    router = build_mlp_router(
        input_size=4,
        output_size=3,
        hidden_size=6,
        use_layer_norm=False,
        activation="relu",
    )
    state = torch.randn(5, 4)

    logits = router(state)

    assert logits.shape == (5, 3)


def test_norm_mlp_uses_layer_norm_when_requested():
    router = build_mlp_router(
        input_size=4,
        output_size=3,
        hidden_size=6,
        use_layer_norm=True,
        activation="gelu",
    )

    assert any(isinstance(module, nn.LayerNorm) for module in router.modules())


def test_stepwise_router_dispatches_by_step():
    first = nn.Linear(4, 3)
    second = nn.Linear(4, 3)
    with torch.no_grad():
        first.bias.fill_(1.0)
        second.bias.fill_(2.0)
    router = StepwiseRouter([first, second])
    state = torch.zeros(2, 4)

    first_logits = router(state, step=0)
    second_logits = router(state, step=1)

    assert first_logits.shape == (2, 3)
    assert second_logits.shape == (2, 3)
    assert not torch.equal(first_logits, second_logits)


def test_input_dependent_routing_full_weights_are_sigmoid_logits():
    torch.manual_seed(1)
    projections = make_projection_set()
    state = torch.randn(2, 4)
    routing = InputDependentRouterRouting(make_stepwise_router())

    plan = routing.plan(state, step=0, projections=projections, mode=router_mode())

    assert plan.selected_indices is None
    assert plan.aggregation_weights.shape == (2, 3)
    assert plan.execution_weights.shape == (2, 3)
    torch.testing.assert_close(
        plan.aggregation_weights,
        torch.sigmoid(plan.aux["router_logits"]),
    )
    assert plan.aux["router_logits"].requires_grad
    assert plan.aux["router_entropy_inputs"] is plan.aux["router_logits"]


def test_uniform_frozen_step_uses_uniform_weights_without_calling_router():
    projections = make_projection_set()
    state = torch.randn(2, 4)
    routing = InputDependentRouterRouting(
        StepwiseRouter([FailingRouter(), nn.Linear(4, 3)]),
        top_k=2,
    )

    frozen_plan = routing.plan(
        state,
        step=0,
        projections=projections,
        mode=uniform_frozen_mode(target_layer=1, top_k=2),
    )
    active_plan = routing.plan(
        state,
        step=1,
        projections=projections,
        mode=uniform_frozen_mode(target_layer=1, top_k=2),
    )

    assert frozen_plan.selected_indices is None
    assert frozen_plan.aggregation_weights.shape == (1, 3)
    assert frozen_plan.execution_weights.shape == (1, 3)
    torch.testing.assert_close(frozen_plan.aggregation_weights, torch.ones(1, 3))
    assert frozen_plan.aux["routing_policy"] == "uniform_frozen"
    assert "router_logits" not in frozen_plan.aux
    assert "router_logits" in active_plan.aux


def test_full_masked_topk_has_exactly_k_nonzero_aggregation_weights_per_sample():
    torch.manual_seed(2)
    projections = make_projection_set()
    state = torch.randn(4, 4)
    routing = InputDependentRouterRouting(make_stepwise_router())

    plan = routing.plan(state, step=0, projections=projections, mode=router_mode(top_k=2))

    assert plan.selected_indices is not None
    assert plan.selected_indices.shape == (4, 2)
    assert plan.aggregation_weights.shape == (4, 3)
    assert torch.equal(
        torch.count_nonzero(plan.aggregation_weights, dim=1),
        torch.full((4,), 2),
    )


def test_full_masked_topk_reports_executed_p_and_aggregated_k():
    torch.manual_seed(3)
    projections = make_projection_set()
    state = torch.randn(2, 4)
    routing = InputDependentRouterRouting(make_stepwise_router())
    plan = routing.plan(state, step=0, projections=projections, mode=router_mode(top_k=2))

    execution = ProjectionExecutor().resolve(projections, plan)

    assert execution.projections is projections
    assert execution.stats["candidate_projections"] == 3
    assert execution.stats["executed_projections"] == 3
    assert execution.stats["aggregated_projections"] == 2
    assert execution.stats["execution_mode"] == "compute_all_mask"


def test_gradients_reach_active_router_parameters_in_decoder_backward_smoke():
    torch.manual_seed(4)
    code = LinearCode(torch.eye(4, dtype=torch.float32))
    projections = make_projection_set()
    stepwise_router = make_stepwise_router()
    routing = InputDependentRouterRouting(stepwise_router)
    decoder = Order2UnfoldedRPADecoder(
        code=code,
        projections=projections,
        bottom_decoder=HadamardOrder1Decoder(length=projections.n // 2),
        routing_policy=routing,
        num_unfolded_steps=1,
        channel_llr=IdentityLLR(),
    )
    channel_output = torch.randn(3, 4, requires_grad=True)

    output = decoder(channel_output, router_mode())
    output.logits.square().mean().backward()

    grads = [parameter.grad for parameter in stepwise_router.routers[0].parameters()]
    assert grads
    assert any(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_collect_debug_false_does_not_put_cpu_tensors_in_stats():
    torch.manual_seed(5)
    code = LinearCode(torch.eye(4, dtype=torch.float32))
    projections = make_projection_set()
    decoder = Order2UnfoldedRPADecoder(
        code=code,
        projections=projections,
        bottom_decoder=HadamardOrder1Decoder(length=projections.n // 2),
        routing_policy=InputDependentRouterRouting(make_stepwise_router()),
        num_unfolded_steps=1,
        channel_llr=IdentityLLR(),
    )

    output = decoder(torch.randn(2, 4), router_mode())

    assert not any(isinstance(value, torch.Tensor) and value.device.type == "cpu" for value in output.stats.values())
    assert "aggregation_weights" not in output.aux["steps"][0]
    assert "execution_weights" not in output.aux["steps"][0]
    assert set(output.aux["steps"][0]["plan_aux"]) == {"router_entropy_inputs"}


def test_collect_debug_true_includes_debug_aux_tensors():
    torch.manual_seed(6)
    code = LinearCode(torch.eye(4, dtype=torch.float32))
    projections = make_projection_set()
    decoder = Order2UnfoldedRPADecoder(
        code=code,
        projections=projections,
        bottom_decoder=HadamardOrder1Decoder(length=projections.n // 2),
        routing_policy=InputDependentRouterRouting(make_stepwise_router()),
        num_unfolded_steps=1,
        channel_llr=IdentityLLR(),
    )
    mode = DecoderMode(
        selection_scope="full",
        execution_mode="compute_all_mask",
        top_k=None,
        forward_depth_policy="full_cascade",
        frozen_policy="frozen_weights",
        collect_debug=True,
    )

    output = decoder(torch.randn(2, 4), mode)

    assert "aggregation_weights" in output.aux["steps"][0]
    assert "execution_weights" in output.aux["steps"][0]
    assert "router_logits" in output.aux["steps"][0]["plan_aux"]
