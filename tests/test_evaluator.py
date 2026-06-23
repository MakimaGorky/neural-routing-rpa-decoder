import pytest
import torch
from torch import nn

from routing_rpa.decoders.modes import DecoderMode
from routing_rpa.decoders.outputs import DecoderOutput
from routing_rpa.eval.evaluator import EvaluationConfig, Evaluator


def eval_mode() -> DecoderMode:
    return DecoderMode(
        selection_scope="full",
        execution_mode="compute_all_mask",
        top_k=None,
        forward_depth_policy="full_cascade",
        frozen_policy="frozen_weights",
    )


class ListStream:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls: list[tuple[int, float]] = []

    def next_batch(self, batch_size: int, snr: float):
        self.calls.append((batch_size, snr))
        return self.batches.pop(0)


class EchoDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.modes: list[DecoderMode] = []

    def forward(self, channel_output: torch.Tensor, mode: DecoderMode) -> DecoderOutput:
        self.modes.append(mode)
        return DecoderOutput(
            logits=channel_output,
            stats={
                "candidate_projections": 3,
                "executed_projections": 3,
                "aggregated_projections": 3,
                "selection_scope": mode.selection_scope,
                "execution_mode": mode.execution_mode,
            },
        )


def test_evaluator_returns_canonical_metrics_counts_and_decoder_stats():
    target = torch.tensor([[0, 1, 1], [1, 0, 0]], dtype=torch.float32)
    logits = torch.tensor([[-1.0, 1.0, 1.0], [1.0, 0.2, -1.0]])
    stream = ListStream([(target, logits)])
    model = EchoDecoder()

    metrics = Evaluator(model, stream).evaluate(
        eval_mode(),
        EvaluationConfig(phase="validation", num_batches=1, batch_size=2, snr=3.5),
    )

    assert metrics["phase"] == "validation"
    assert metrics["ber"] == pytest.approx(1 / 6)
    assert metrics["bler"] == pytest.approx(1 / 2)
    assert metrics["bit_errors"] == 1
    assert metrics["block_errors"] == 1
    assert metrics["total_bits"] == 6
    assert metrics["total_words"] == 2
    assert metrics["correct_words"] == 1
    assert metrics["candidate_projections"] == 3
    assert metrics["executed_projections"] == 3
    assert metrics["aggregated_projections"] == 3
    assert stream.calls == [(2, 3.5)]


def test_evaluator_uses_bit_weighted_ber_across_batches():
    first_target = torch.tensor([[0, 0]], dtype=torch.float32)
    first_logits = torch.tensor([[1.0, -1.0]])
    second_target = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    second_logits = torch.tensor([[1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
    stream = ListStream(
        [
            (first_target, first_logits),
            (second_target, second_logits),
        ]
    )

    metrics = Evaluator(EchoDecoder(), stream).evaluate(
        eval_mode(),
        EvaluationConfig(phase="final", num_batches=2, batch_size=1, snr=1.0),
    )

    assert metrics["bit_errors"] == 2
    assert metrics["total_bits"] == 8
    assert metrics["ber"] == pytest.approx(2 / 8)
    assert metrics["ber"] != pytest.approx(((1 / 2) + (1 / 6)) / 2)


def test_evaluator_runs_model_in_explicit_mode_and_restores_training_state():
    target = torch.tensor([[0, 1]], dtype=torch.float32)
    logits = torch.tensor([[-1.0, 1.0]])
    model = EchoDecoder()
    model.train(True)
    mode = eval_mode()

    Evaluator(model, ListStream([(target, logits)])).evaluate(
        mode,
        EvaluationConfig(phase="validation", num_batches=1, batch_size=1, snr=0.0),
    )

    assert model.training is True
    assert model.modes[0].selection_scope == mode.selection_scope
    assert model.modes[0].channel_context == {"snr": 0.0}


def test_evaluator_passes_config_snr_to_decoder_mode():
    target = torch.tensor([[0, 1]], dtype=torch.float32)
    logits = torch.tensor([[-1.0, 1.0]])
    model = EchoDecoder()

    Evaluator(model, ListStream([(target, logits)])).evaluate(
        eval_mode(),
        EvaluationConfig(phase="validation", num_batches=1, batch_size=1, snr=4.25),
    )

    assert model.modes[0].channel_context["snr"] == 4.25
