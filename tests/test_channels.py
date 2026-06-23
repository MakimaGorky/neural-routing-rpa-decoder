import inspect
import math

import pytest
import torch

from routing_rpa.channels.awgn import AWGNChannel, AWGNLLR, FixedSnrAWGNLLR, bpsk_modulate
from routing_rpa.channels.fading import AWGNFadingChannel
from routing_rpa.channels.reliability_profiles import CoordinateReliabilityAWGN
from routing_rpa.codes.linear import LinearCode


def make_code() -> LinearCode:
    return LinearCode(
        torch.tensor(
            [
                [1, 1, 1, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
            ],
            dtype=torch.float32,
        )
    )


def test_sigma_formula_matches_legacy_formula():
    code = make_code()
    snr = 4.0
    expected_sigma = math.sqrt(1 / (2 * (code.k / code.n) * 10 ** (snr / 10)))

    channel = AWGNChannel()

    assert channel.sigma(snr, code) == pytest.approx(expected_sigma)
    assert channel.sigma_sqr(snr, code) == pytest.approx(expected_sigma**2)


def test_bpsk_modulation_preserves_legacy_convention():
    bits = torch.tensor([[0, 1, 1, 0]], dtype=torch.float32)

    modulated = bpsk_modulate(bits)

    torch.testing.assert_close(
        modulated,
        torch.tensor([[1.0, -1.0, -1.0, 1.0]], dtype=torch.float32),
    )


def test_awgn_output_shape_equals_input_codeword_shape():
    torch.manual_seed(0)
    code = make_code()
    channel = AWGNChannel()
    bits = torch.randint(0, 2, (5, code.n), dtype=torch.float32)

    output = channel(bits, snr=3.0, code=code)

    assert output.shape == (5, code.n)
    assert output.dtype == torch.float32


def test_awgn_llr_uses_explicit_sigma_sqr_component():
    code = make_code()
    llr = AWGNLLR()
    received = torch.tensor([[1.0, -2.0, 0.5, 0.0]], dtype=torch.float32)

    converted = llr(received, snr=1.5, code=code)

    torch.testing.assert_close(converted, 2.0 * received / llr.sigma_sqr(1.5, code))


def test_fixed_snr_awgn_llr_is_decoder_compatible():
    code = make_code()
    received = torch.tensor([[1.0, -2.0, 0.5, 0.0]], dtype=torch.float32)
    fixed = FixedSnrAWGNLLR(snr=1.5)

    converted = fixed(received, code)

    torch.testing.assert_close(converted, AWGNLLR()(received, 1.5, code))


def test_fixed_snr_awgn_llr_accepts_architectural_signature():
    code = make_code()
    received = torch.tensor([[1.0, -2.0, 0.5, 0.0]], dtype=torch.float32)
    fixed = FixedSnrAWGNLLR(snr=1.5)

    converted = fixed(received, snr=-99.0, code=code)

    torch.testing.assert_close(converted, AWGNLLR()(received, 1.5, code))


def test_awgn_fading_channel_shape_for_block_and_coordinate_fading():
    torch.manual_seed(1)
    code = make_code()
    bits = torch.randint(0, 2, (3, code.n), dtype=torch.float32)

    coordinate_output = AWGNFadingChannel(block_fading=False)(bits, snr=2.0, code=code)
    block_output = AWGNFadingChannel(block_fading=True)(bits, snr=2.0, code=code)

    assert coordinate_output.shape == bits.shape
    assert block_output.shape == bits.shape


def test_prefix_reliability_profile_changes_prefix_noise_scale():
    code = make_code()
    channel = CoordinateReliabilityAWGN(prefix_len=2, noise_multiplier=3.0)

    std = channel.noise_std(
        snr=2.0,
        code=code,
        shape=(4, code.n),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    base_sigma = channel.sigma(2.0, code)
    assert std.shape == (1, code.n)
    torch.testing.assert_close(std[0, :2], torch.full((2,), base_sigma * 3.0))
    torch.testing.assert_close(std[0, 2:], torch.full((code.n - 2,), base_sigma))


def test_coordinate_reliability_uses_prefix_len_not_legacy_k_name():
    signature = inspect.signature(CoordinateReliabilityAWGN)

    assert "prefix_len" in signature.parameters
    assert "k" not in signature.parameters
