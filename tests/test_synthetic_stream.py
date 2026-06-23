import torch
import pytest

from routing_rpa.channels.awgn import AWGNChannel
from routing_rpa.channels.reliability_profiles import CoordinateReliabilityAWGN
from routing_rpa.codes.linear import LinearCode
from routing_rpa.data.synthetic import SyntheticCodewordStream


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


def test_synthetic_stream_returns_target_bits_and_channel_output_shapes():
    torch.manual_seed(2)
    code = make_code()
    stream = SyntheticCodewordStream(code, AWGNChannel())

    target_bits, channel_output = stream.next_batch(batch_size=6, snr=3.0)

    assert target_bits.shape == (6, code.n)
    assert channel_output.shape == (6, code.n)


def test_synthetic_stream_target_bits_are_binary():
    torch.manual_seed(3)
    code = make_code()
    stream = SyntheticCodewordStream(code, AWGNChannel())

    target_bits, _ = stream.next_batch(batch_size=8, snr=1.0)

    assert torch.all((target_bits == 0) | (target_bits == 1))


def test_synthetic_stream_accepts_coordinate_reliability_channel():
    torch.manual_seed(4)
    code = make_code()
    stream = SyntheticCodewordStream(
        code,
        CoordinateReliabilityAWGN(prefix_len=2, noise_multiplier=2.0),
    )

    target_bits, channel_output = stream.next_batch(batch_size=3, snr=2.5)

    assert target_bits.shape == channel_output.shape == (3, code.n)


def test_synthetic_stream_rejects_device_mismatch_before_silent_cpu_fallback():
    code = make_code()

    with pytest.raises(ValueError, match="device must match"):
        SyntheticCodewordStream(code, AWGNChannel(), device="cuda")
