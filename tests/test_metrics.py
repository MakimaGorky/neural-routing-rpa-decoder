import pytest
import torch

from routing_rpa.training.metrics import (
    BinaryMetricCounts,
    compute_binary_metrics,
    compute_metrics_from_logits,
    logits_to_bits,
    merge_metric_counts,
)


def test_metrics_use_canonical_names_without_legacy_true_ber():
    target = torch.tensor([[0, 1, 1], [1, 0, 0]], dtype=torch.int64)
    predicted = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.int64)

    metrics = compute_binary_metrics(predicted, target)

    assert set(metrics) == {
        "ber",
        "bler",
        "correct_words",
        "total_words",
        "total_bits",
        "bit_errors",
        "block_errors",
    }
    assert "true_ber" not in metrics


def test_ber_is_total_bit_errors_over_total_bits():
    target = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64)
    predicted = torch.tensor([[1, 0], [1, 1]], dtype=torch.int64)

    metrics = compute_binary_metrics(predicted, target)

    assert metrics["bit_errors"] == 3
    assert metrics["total_bits"] == 4
    assert metrics["ber"] == pytest.approx(3 / 4)


def test_bler_is_block_errors_over_total_words():
    target = torch.tensor([[0, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.int64)
    predicted = torch.tensor([[0, 0, 0], [1, 0, 1], [1, 0, 0]], dtype=torch.int64)

    metrics = compute_binary_metrics(predicted, target)

    assert metrics["block_errors"] == 2
    assert metrics["correct_words"] == 1
    assert metrics["total_words"] == 3
    assert metrics["bler"] == pytest.approx(2 / 3)


def test_accumulated_ber_is_bit_weighted_not_average_batch_ber():
    first = BinaryMetricCounts()
    first.update(
        predicted_bits=torch.tensor([[1, 0]], dtype=torch.int64),
        target_bits=torch.tensor([[0, 0]], dtype=torch.int64),
    )
    second = BinaryMetricCounts()
    second.update(
        predicted_bits=torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.int64),
        target_bits=torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.int64),
    )

    metrics = merge_metric_counts([first, second]).compute()

    assert first.rates()["ber"] == pytest.approx(1 / 2)
    assert second.rates()["ber"] == pytest.approx(1 / 6)
    assert metrics["ber"] == pytest.approx(2 / 8)
    assert metrics["ber"] != pytest.approx(((1 / 2) + (1 / 6)) / 2)


def test_logits_to_bits_and_metrics_from_logits_use_positive_as_bit_one():
    logits = torch.tensor([[-1.0, 0.25, 2.0], [0.0, -0.1, 0.1]])
    target = torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.int64)

    predicted = logits_to_bits(logits)
    metrics = compute_metrics_from_logits(logits, target)

    torch.testing.assert_close(
        predicted,
        torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.int64),
    )
    assert metrics["ber"] == 0.0
    assert metrics["bler"] == 0.0
