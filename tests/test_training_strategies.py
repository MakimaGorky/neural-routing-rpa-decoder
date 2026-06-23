import pytest
import torch
from torch import nn

from routing_rpa.decoders.routing import InputDependentRouterRouting, StepwiseRouter
from routing_rpa.training.strategies import LayerwiseTraining


class TinyLayeredModel(nn.Module):
    def __init__(self, *, num_layers: int = 3) -> None:
        super().__init__()
        self.num_unfolded_steps = num_layers
        routers = [nn.Linear(4, 2) for _ in range(num_layers)]
        self.routing_policy = InputDependentRouterRouting(StepwiseRouter(routers))


def parameter_ids(parameters):
    return {id(parameter) for parameter in parameters}


def test_layerwise_strategy_returns_all_or_configured_layer_ids():
    model = TinyLayeredModel(num_layers=3)

    assert LayerwiseTraining().layers_to_train(model) == [0, 1, 2]
    assert LayerwiseTraining(train_layers=[2, 0]).layers_to_train(model) == [2, 0]


def test_layerwise_strategy_rejects_duplicate_or_out_of_range_layers():
    model = TinyLayeredModel(num_layers=2)

    with pytest.raises(ValueError, match="duplicate"):
        LayerwiseTraining(train_layers=[0, 0]).layers_to_train(model)

    with pytest.raises(ValueError, match="outside available layers"):
        LayerwiseTraining(train_layers=[2]).layers_to_train(model)


def test_layerwise_trainable_parameters_only_include_active_router_layer():
    model = TinyLayeredModel(num_layers=3)
    strategy = LayerwiseTraining()

    active = list(strategy.trainable_parameters(model, layer_id=1))

    active_ids = parameter_ids(active)
    layer0_ids = parameter_ids(model.routing_policy.router.routers[0].parameters())
    layer1_ids = parameter_ids(model.routing_policy.router.routers[1].parameters())
    layer2_ids = parameter_ids(model.routing_policy.router.routers[2].parameters())

    assert active_ids == layer1_ids
    assert active_ids.isdisjoint(layer0_ids)
    assert active_ids.isdisjoint(layer2_ids)
    assert all(parameter.requires_grad for parameter in model.routing_policy.router.routers[1].parameters())
    assert not any(parameter.requires_grad for parameter in model.routing_policy.router.routers[0].parameters())
    assert not any(parameter.requires_grad for parameter in model.routing_policy.router.routers[2].parameters())


def test_layerwise_mode_contains_explicit_depth_target_and_frozen_policy():
    strategy = LayerwiseTraining(
        top_k=2,
        forward_depth_policy="local_depth",
        frozen_policy="no_grad_frozen",
        collect_debug=True,
    )

    mode = strategy.mode_for_step(layer_id=1, phase="train")

    assert mode.selection_scope == "full"
    assert mode.execution_mode == "compute_all_mask"
    assert mode.top_k == 2
    assert mode.forward_depth_policy == "local_depth"
    assert mode.frozen_policy == "no_grad_frozen"
    assert mode.target_layer == 1
    assert mode.collect_debug is True


def test_skip_future_policy_uses_local_depth_even_when_full_cascade_is_configured():
    strategy = LayerwiseTraining(
        forward_depth_policy="full_cascade",
        frozen_policy="skip_future",
    )

    mode = strategy.mode_for_step(layer_id=2, phase="train")

    assert mode.forward_depth_policy == "local_depth"
    assert mode.target_layer == 2


def test_layerwise_strategy_requires_explicit_layer_for_trainable_parameters():
    model = TinyLayeredModel(num_layers=1)

    with pytest.raises(ValueError, match="explicit layer_id"):
        list(LayerwiseTraining().trainable_parameters(model, layer_id=None))
