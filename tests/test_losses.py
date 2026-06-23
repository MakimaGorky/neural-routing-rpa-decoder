import pytest
import torch
import torch.nn.functional as F

from routing_rpa.decoders.outputs import DecoderOutput
from routing_rpa.training.losses import LossComposer, LossConfig


def test_bce_loss_differentiates_through_logits():
    logits = torch.tensor([[0.5, -1.0, 1.5]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 1.0]])
    composer = LossComposer(
        LossConfig(components=[{"name": "bce_logits", "weight": 1.0}])
    )

    loss, scalars = composer(DecoderOutput(logits=logits), target)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert scalars["bce_logits"] == pytest.approx(
        F.binary_cross_entropy_with_logits(logits.detach(), target).item()
    )
    assert "total" in scalars


def test_bce_and_soft_ber_prefer_positive_logits_for_bit_one():
    target = torch.tensor([[1.0, 0.0]])
    good_logits = torch.tensor([[2.0, -2.0]])
    bad_logits = -good_logits

    bce = LossComposer(LossConfig(components=[{"name": "bce_logits", "weight": 1.0}]))
    soft_ber = LossComposer(
        LossConfig(components=[{"name": "soft_ber", "weight": 1.0, "beta": 3.0}])
    )

    good_bce, _ = bce(DecoderOutput(logits=good_logits), target)
    bad_bce, _ = bce(DecoderOutput(logits=bad_logits), target)
    good_soft_ber, _ = soft_ber(DecoderOutput(logits=good_logits), target)
    bad_soft_ber, _ = soft_ber(DecoderOutput(logits=bad_logits), target)

    assert good_bce < bad_bce
    assert good_soft_ber < bad_soft_ber


def test_zero_weight_components_do_not_affect_total_loss_or_require_aux():
    logits = torch.tensor([[0.2, -0.4, 0.7]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 1.0]])
    output = DecoderOutput(logits=logits)
    bce_only = LossComposer(
        LossConfig(components=[{"name": "bce_logits", "weight": 1.0}])
    )
    with_zero_weight_terms = LossComposer(
        LossConfig(
            components=[
                {"name": "bce_logits", "weight": 1.0},
                {"name": "soft_ber", "weight": 0.0},
                {"name": "router_entropy", "weight": 0.0},
            ]
        )
    )

    base_loss, _ = bce_only(output, target)
    combined_loss, scalars = with_zero_weight_terms(output, target)

    torch.testing.assert_close(combined_loss, base_loss)
    assert scalars["soft_ber_weighted"] == 0.0
    assert scalars["router_entropy_weighted"] == 0.0


def test_router_entropy_contributes_and_keeps_gradient_path():
    logits = torch.tensor([[0.1, -0.2, 0.3]], requires_grad=True)
    router_logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
    output = DecoderOutput(
        logits=logits,
        aux={"steps": [{"plan_aux": {"router_entropy_inputs": router_logits}}]},
    )
    target = torch.tensor([[1.0, 0.0, 1.0]])
    composer = LossComposer(
        LossConfig(
            components=[
                {"name": "bce_logits", "weight": 0.0},
                {"name": "router_entropy", "weight": 0.5},
            ]
        )
    )

    loss, scalars = composer(output, target)
    loss.backward()

    assert scalars["router_entropy"] > 0.0
    assert scalars["router_entropy_weighted"] > 0.0
    assert router_logits.grad is not None
    assert torch.isfinite(router_logits.grad).all()


def test_loss_scalars_are_returned_with_stable_names():
    logits = torch.tensor([[0.1, -0.2]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0]])
    composer = LossComposer(
        LossConfig(
            components=[
                {"name": "bce_logits", "weight": 1.0},
                {"name": "soft_ber", "weight": 0.25, "beta": 3.0},
                {"name": "legacy_elementary_ce", "weight": 0.1},
            ]
        )
    )

    _, scalars = composer(DecoderOutput(logits=logits), target)

    assert set(scalars) == {
        "bce_logits",
        "bce_logits_weighted",
        "soft_ber",
        "soft_ber_weighted",
        "legacy_elementary_ce",
        "legacy_elementary_ce_weighted",
        "total",
    }


def test_router_entropy_requires_aux_when_weight_is_positive():
    logits = torch.tensor([[0.1, -0.2]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0]])
    composer = LossComposer(
        LossConfig(components=[{"name": "router_entropy", "weight": 1.0}])
    )

    with pytest.raises(ValueError, match="router_entropy_inputs"):
        composer(DecoderOutput(logits=logits), target)
