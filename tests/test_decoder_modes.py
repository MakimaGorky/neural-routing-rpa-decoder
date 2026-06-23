import pytest
import torch

from routing_rpa.decoders.modes import DecoderMode
from routing_rpa.decoders.outputs import DecoderOutput
from routing_rpa.decoders.selection import ProjectionPlan


def test_valid_decoder_mode_constructs_successfully():
    mode = DecoderMode(
        selection_scope="full",
        execution_mode="compute_all_mask",
        top_k=None,
        forward_depth_policy="full_cascade",
        frozen_policy="frozen_weights",
    )

    assert mode.selection_scope == "full"
    assert mode.execution_mode == "compute_all_mask"
    assert mode.resolve_num_steps(5) == 5
    assert dict(mode.channel_context) == {}


def test_decoder_mode_carries_explicit_channel_context():
    mode = DecoderMode(
        selection_scope="full",
        execution_mode="compute_all_mask",
        top_k=None,
        forward_depth_policy="full_cascade",
        frozen_policy="frozen_weights",
    )

    with_snr = mode.with_channel_context(snr=2.5)

    assert dict(mode.channel_context) == {}
    assert dict(with_snr.channel_context) == {"snr": 2.5}


def test_decoder_mode_channel_context_is_immutable_snapshot():
    context = {"snr": 1.0}
    mode = DecoderMode(
        selection_scope="full",
        execution_mode="compute_all_mask",
        top_k=None,
        forward_depth_policy="full_cascade",
        frozen_policy="frozen_weights",
        channel_context=context,
    )

    context["snr"] = 9.0
    context["extra"] = "changed"

    assert dict(mode.channel_context) == {"snr": 1.0}
    with pytest.raises(TypeError):
        mode.channel_context["snr"] = 2.0


def test_invalid_literal_like_values_fail_early():
    with pytest.raises(ValueError, match="selection_scope"):
        DecoderMode(
            selection_scope="everything",
            execution_mode="compute_all_mask",
            top_k=None,
            forward_depth_policy="full_cascade",
            frozen_policy="frozen_weights",
        )

    with pytest.raises(ValueError, match="execution_mode"):
        DecoderMode(
            selection_scope="full",
            execution_mode="mask_somehow",
            top_k=None,
            forward_depth_policy="full_cascade",
            frozen_policy="frozen_weights",
        )

    with pytest.raises(ValueError, match="frozen_policy"):
        DecoderMode(
            selection_scope="full",
            execution_mode="compute_all_mask",
            top_k=None,
            forward_depth_policy="full_cascade",
            frozen_policy="freeze_magic",
        )


def test_local_depth_requires_target_layer_and_resolves_depth():
    with pytest.raises(ValueError, match="local_depth requires target_layer"):
        DecoderMode(
            selection_scope="full",
            execution_mode="compute_all_mask",
            top_k=None,
            forward_depth_policy="local_depth",
            frozen_policy="frozen_weights",
        )

    mode = DecoderMode(
        selection_scope="full",
        execution_mode="compute_all_mask",
        top_k=None,
        forward_depth_policy="local_depth",
        frozen_policy="skip_future",
        target_layer=2,
    )
    assert mode.resolve_num_steps(5) == 3
    assert mode.resolve_num_steps(2) == 2


def test_skip_future_policy_limits_depth_even_with_full_cascade_policy():
    mode = DecoderMode(
        selection_scope="full",
        execution_mode="compute_all_mask",
        top_k=None,
        forward_depth_policy="full_cascade",
        frozen_policy="skip_future",
        target_layer=1,
    )

    assert mode.resolve_num_steps(5) == 2
    assert mode.resolve_num_steps(1) == 1


def test_frozen_policy_step_helpers_are_explicit():
    mode = DecoderMode(
        selection_scope="full",
        execution_mode="compute_all_mask",
        top_k=None,
        forward_depth_policy="local_depth",
        frozen_policy="uniform_frozen",
        target_layer=2,
    )

    assert mode.is_frozen_step(0) is True
    assert mode.is_frozen_step(2) is False
    assert mode.uses_uniform_frozen_weights(0) is True
    assert mode.uses_uniform_frozen_weights(2) is False

    no_grad = DecoderMode(
        selection_scope="full",
        execution_mode="compute_all_mask",
        top_k=None,
        forward_depth_policy="local_depth",
        frozen_policy="no_grad_frozen",
        target_layer=2,
    )
    assert no_grad.runs_no_grad(0) is True
    assert no_grad.runs_no_grad(1) is True
    assert no_grad.runs_no_grad(2) is False
    assert no_grad.runs_no_grad(3) is True


def test_decoder_output_preserves_aux_and_stats_contract():
    logits = torch.zeros(2, 4)
    output = DecoderOutput(
        logits=logits,
        aux={"router_logits": torch.ones(2, 3)},
        stats={"candidate_projections": 3},
    )

    assert output.logits.shape == (2, 4)
    assert output.aux["router_logits"].shape == (2, 3)
    assert output.stats["candidate_projections"] == 3


def test_projection_plan_validates_shapes_and_counts():
    plan = ProjectionPlan(
        candidate_count=4,
        selected_indices=None,
        selection_scope="full",
        execution_mode="compute_all_mask",
        aggregation_weights=torch.ones(1, 4),
        execution_weights=torch.ones(1, 4),
    )

    assert plan.aggregated_count == 4
    assert plan.execution_count == 4

    with pytest.raises(ValueError, match="candidate_count"):
        ProjectionPlan(
            candidate_count=0,
            selected_indices=None,
            selection_scope="full",
            execution_mode="compute_all_mask",
            aggregation_weights=torch.ones(1, 4),
            execution_weights=torch.ones(1, 4),
        )

    with pytest.raises(ValueError, match="aggregation_weights"):
        ProjectionPlan(
            candidate_count=4,
            selected_indices=None,
            selection_scope="full",
            execution_mode="compute_all_mask",
            aggregation_weights=torch.ones(4),
            execution_weights=torch.ones(1, 4),
        )
