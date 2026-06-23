from pathlib import Path

import pytest
import torch

from routing_rpa.decoders.bottom_decoders import build_hadamard_matrix
from routing_rpa.decoders.kernels_order2 import (
    aggregate_1d,
    convert_to_llr,
    hadamard_decode_order1,
    project_1d,
)
from routing_rpa.projections.loaders import load_legacy_projection_file
from routing_rpa.projections.projection_set import ProjectionSet


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PROJECTIONS_TXT = REPO_ROOT / "src_old" / "data" / "decoder" / "projections.txt"


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


def test_convert_to_llr_uses_explicit_sigma_sqr():
    received = torch.tensor([[1.0, -2.0, 0.5, 0.0]])

    converted = convert_to_llr(received, sigma_sqr=0.25)

    torch.testing.assert_close(converted, 2.0 * received / 0.25)


def test_convert_to_llr_rejects_non_positive_sigma_sqr():
    with pytest.raises(ValueError, match="sigma_sqr"):
        convert_to_llr(torch.ones(1, 4), sigma_sqr=0.0)


def test_project_1d_shape_on_small_synthetic_projections():
    projections = make_projection_set()
    llr = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    projected = project_1d(llr, projections)

    assert projected.shape == (2, 3, 2)


def test_project_1d_matches_log_domain_soft_xor_formula():
    projections = make_projection_set()
    llr = torch.tensor([[0.5, -1.0, 2.0, -0.25]], dtype=torch.float32)

    projected = project_1d(llr, projections)

    first = llr[:, torch.tensor([[0, 1], [0, 2], [0, 3]])]
    second = llr[:, torch.tensor([[2, 3], [1, 3], [1, 2]])]
    expected = torch.nn.functional.softplus(first + second) - torch.logaddexp(first, second)
    torch.testing.assert_close(projected, expected)


def test_project_1d_shape_on_real_legacy_projection_set():
    projections = load_legacy_projection_file(LEGACY_PROJECTIONS_TXT, expected_count=512)
    llr = torch.zeros(2, projections.n)

    projected = project_1d(llr, projections)

    assert projected.shape == (2, 512, 512)


def test_hadamard_decode_order1_preserves_shape():
    projected = torch.randn(2, 3, 4)
    H = build_hadamard_matrix(4)

    decoded = hadamard_decode_order1(projected, H)

    assert decoded.shape == projected.shape


def test_aggregate_1d_returns_length_n_with_shared_weights():
    projections = make_projection_set()
    received = torch.randn(2, projections.n)
    decoded = torch.randn(2, projections.num_projections, projections.n // 2)
    weights = torch.ones(1, projections.num_projections)

    aggregated = aggregate_1d(received, decoded, weights, projections)

    assert aggregated.shape == (2, projections.n)
    assert torch.isfinite(aggregated).all()


def test_aggregate_1d_accepts_per_batch_weights():
    projections = make_projection_set()
    received = torch.randn(2, projections.n)
    decoded = torch.randn(2, projections.num_projections, projections.n // 2)
    weights = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    aggregated = aggregate_1d(received, decoded, weights, projections)

    assert aggregated.shape == (2, projections.n)
    assert torch.isfinite(aggregated).all()


def test_aggregate_1d_debug_norm_check_catches_zero_normalization():
    projections = make_projection_set()
    received = torch.randn(2, projections.n)
    decoded = torch.randn(2, projections.num_projections, projections.n // 2)
    weights = torch.zeros(1, projections.num_projections)

    with pytest.raises(ValueError, match="normalization"):
        aggregate_1d(received, decoded, weights, projections, check_norm=True)


def test_cpu_backward_through_project_hadamard_and_aggregate():
    projections = make_projection_set()
    received = torch.randn(2, projections.n, requires_grad=True)
    weights = torch.ones(1, projections.num_projections, requires_grad=True)
    H = build_hadamard_matrix(projections.n // 2)

    projected = project_1d(received, projections)
    decoded = hadamard_decode_order1(projected, H)
    aggregated = aggregate_1d(received, decoded, weights, projections)
    loss = aggregated.square().mean()
    loss.backward()

    assert received.grad is not None
    assert weights.grad is not None
    assert torch.isfinite(received.grad).all()
    assert torch.isfinite(weights.grad).all()
