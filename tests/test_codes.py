from pathlib import Path

import pytest
import torch

from routing_rpa.codes.gf2 import is_binary_tensor
from routing_rpa.codes.linear import LinearCode
from routing_rpa.codes.reed_muller import RMCode, rm_dimension


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_G_TXT = REPO_ROOT / "src_old" / "data" / "decoder" / "G.txt"


def test_linear_code_rejects_non_binary_generator_matrix():
    with pytest.raises(ValueError, match="binary"):
        LinearCode(torch.tensor([[0, 1, 2]], dtype=torch.float32))


def test_linear_code_encode_matches_manual_mod2_multiplication():
    generator = torch.tensor(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=torch.float32,
    )
    messages = torch.tensor(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=torch.float32,
    )
    code = LinearCode(generator)

    encoded = code.encode(messages)

    expected = torch.tensor(
        [
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    assert encoded.shape == (3, 4)
    assert encoded.dtype == torch.float32
    torch.testing.assert_close(encoded, expected)


def test_linear_code_encode_validates_message_shape_and_bits():
    code = LinearCode(torch.eye(2, dtype=torch.float32))

    with pytest.raises(ValueError, match="columns"):
        code.encode(torch.tensor([[1, 0, 1]], dtype=torch.float32))

    with pytest.raises(ValueError, match="binary"):
        code.encode(torch.tensor([[1, 0.5]], dtype=torch.float32))


def test_rm_dimension_matches_rm_10_2_contract():
    assert rm_dimension(10, 2) == 56


def test_rm_code_loads_legacy_rm_10_2_without_current_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    code = RMCode.from_text_file(LEGACY_G_TXT)

    assert code.m == 10
    assert code.r == 2
    assert code.n == 1024
    assert code.k == 56
    assert code.generator_matrix.shape == (56, 1024)
    assert is_binary_tensor(code.generator_matrix)


def test_rm_code_can_load_tensor_artifact(tmp_path):
    generator = torch.tensor(
        [
            [1, 1, 1, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    artifact = tmp_path / "rm_2_1_G.pt"
    torch.save(generator, artifact)

    code = RMCode.from_artifact(artifact)

    assert code.m == 2
    assert code.r == 1
    assert code.n == 4
    assert code.k == 3
    torch.testing.assert_close(code.generator_matrix, generator)


def test_rm_code_encoded_output_is_binary():
    code = RMCode.from_text_file(LEGACY_G_TXT)
    messages = torch.randint(0, 2, (4, code.k), dtype=torch.int64).to(code.dtype)

    encoded = code.encode(messages)

    assert encoded.shape == (4, 1024)
    assert encoded.dtype == code.dtype
    assert is_binary_tensor(encoded)
