import pytest
import torch

from routing_rpa.codes.gf2 import (
    gf2_matmul,
    gf2_rank,
    gf2_row_reduce,
    has_full_row_rank,
    is_binary_tensor,
    validate_binary_tensor,
)


def test_binary_validation_accepts_only_zero_one_entries():
    assert is_binary_tensor(torch.tensor([[0, 1], [1, 0]], dtype=torch.float32))
    assert is_binary_tensor(torch.tensor([[False, True], [True, False]]))
    assert not is_binary_tensor(torch.tensor([[0, 2], [1, 0]]))

    with pytest.raises(ValueError, match="binary"):
        validate_binary_tensor(torch.tensor([[0.0, 0.5]]), name="bad_matrix")


def test_gf2_matmul_matches_manual_mod2_result():
    left = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.float32)
    right = torch.tensor(
        [
            [1, 1],
            [0, 1],
            [1, 0],
        ],
        dtype=torch.float32,
    )

    result = gf2_matmul(left, right)

    expected = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gf2_matmul_runs_on_cuda():
    left = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.float32, device="cuda")
    right = torch.tensor(
        [
            [1, 1],
            [0, 1],
            [1, 0],
        ],
        dtype=torch.float32,
        device="cuda",
    )

    result = gf2_matmul(left, right)

    expected = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32, device="cuda")
    torch.testing.assert_close(result, expected)


def test_gf2_rank_on_known_small_matrices():
    zero = torch.zeros((3, 4), dtype=torch.uint8)
    identity = torch.eye(3, dtype=torch.uint8)
    dependent = torch.tensor(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=torch.uint8,
    )

    assert gf2_rank(zero) == 0
    assert gf2_rank(identity) == 3
    assert gf2_rank(dependent) == 2


def test_duplicate_rows_reduce_gf2_rank():
    matrix = torch.tensor(
        [
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
        ],
        dtype=torch.uint8,
    )

    assert gf2_rank(matrix) == 2
    assert not has_full_row_rank(matrix)


def test_gf2_row_reduce_returns_pivot_columns():
    matrix = torch.tensor(
        [
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
        ],
        dtype=torch.uint8,
    )

    reduced, pivots = gf2_row_reduce(matrix)

    assert pivots == [0, 1]
    assert reduced.dtype == torch.uint8
    assert gf2_rank(reduced) == 2
