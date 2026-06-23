import pytest
import torch

from routing_rpa.decoders.bottom_decoders import (
    BottomDecoder,
    HadamardOrder1Decoder,
    build_hadamard_matrix,
)
from routing_rpa.projections.projection_set import ProjectionSet


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


def test_bottom_decoder_base_is_abstract():
    with pytest.raises(TypeError):
        BottomDecoder()


def test_build_hadamard_matrix_has_expected_sylvester_values():
    H = build_hadamard_matrix(4)

    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ]
    )
    torch.testing.assert_close(H, expected)
    torch.testing.assert_close(H @ H, torch.eye(4) * 4)


def test_build_hadamard_matrix_rejects_non_power_of_two_length():
    with pytest.raises(ValueError, match="power of two"):
        build_hadamard_matrix(6)


def test_hadamard_order1_decoder_registers_non_trainable_buffer():
    decoder = HadamardOrder1Decoder(2)

    assert "hadamard_matrix" in dict(decoder.named_buffers())
    assert dict(decoder.named_parameters()) == {}
    assert decoder.hadamard_matrix.requires_grad is False


def test_hadamard_order1_decoder_preserves_shape():
    projections = make_projection_set()
    decoder = HadamardOrder1Decoder(length=projections.n // 2)
    projected = torch.randn(2, projections.num_projections, projections.n // 2)

    decoded = decoder(projected, projections, step=0)

    assert decoded.shape == projected.shape


def test_hadamard_order1_decoder_backward_on_cpu():
    projections = make_projection_set()
    decoder = HadamardOrder1Decoder(length=projections.n // 2)
    projected = torch.randn(
        2,
        projections.num_projections,
        projections.n // 2,
        requires_grad=True,
    )

    decoded = decoder(projected, projections, step=1)
    decoded.square().sum().backward()

    assert projected.grad is not None
    assert torch.isfinite(projected.grad).all()


def test_hadamard_order1_decoder_rejects_projection_count_mismatch():
    projections = make_projection_set()
    decoder = HadamardOrder1Decoder(length=projections.n // 2)
    projected = torch.randn(2, projections.num_projections - 1, projections.n // 2)

    with pytest.raises(ValueError, match="projection count"):
        decoder(projected, projections, step=0)
