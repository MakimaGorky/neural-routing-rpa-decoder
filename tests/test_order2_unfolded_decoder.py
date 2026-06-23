from pathlib import Path

import torch

from routing_rpa.channels.awgn import AWGNLLR
from routing_rpa.codes.linear import LinearCode
from routing_rpa.codes.reed_muller import RMCode
from routing_rpa.decoders.bottom_decoders import HadamardOrder1Decoder
from routing_rpa.decoders.modes import DecoderMode
from routing_rpa.decoders.order2_unfolded import (
    Order2UnfoldedRPADecoder,
    resolve_unfolded_steps,
)
from routing_rpa.decoders.routing import (
    InputDependentRouterRouting,
    StepwiseRouter,
    UniformRouting,
)
from routing_rpa.projections.loaders import load_legacy_projection_file
from routing_rpa.projections.projection_set import ProjectionSet


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_G_TXT = REPO_ROOT / "src_old" / "data" / "decoder" / "G.txt"
LEGACY_PROJECTIONS_TXT = REPO_ROOT / "src_old" / "data" / "decoder" / "projections.txt"


class IdentityLLR:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, received: torch.Tensor, code: LinearCode) -> torch.Tensor:
        self.calls += 1
        return received


class CountingUniformRouting(UniformRouting):
    def __init__(self) -> None:
        super().__init__()
        self.plan_calls = 0

    def plan(self, state, step, projections, mode):
        self.plan_calls += 1
        return super().plan(state, step, projections, mode)


def make_toy_projection_set() -> ProjectionSet:
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


def uniform_full_mode() -> DecoderMode:
    return DecoderMode(
        selection_scope="full",
        execution_mode="compute_all_mask",
        top_k=None,
        forward_depth_policy="full_cascade",
        frozen_policy="frozen_weights",
    )


def local_depth_mode(target_layer: int) -> DecoderMode:
    return DecoderMode(
        selection_scope="full",
        execution_mode="compute_all_mask",
        top_k=None,
        forward_depth_policy="local_depth",
        frozen_policy="frozen_weights",
        target_layer=target_layer,
    )


def no_grad_frozen_mode(target_layer: int) -> DecoderMode:
    return DecoderMode(
        selection_scope="full",
        execution_mode="compute_all_mask",
        top_k=None,
        forward_depth_policy="local_depth",
        frozen_policy="no_grad_frozen",
        target_layer=target_layer,
    )


def make_toy_decoder(*, num_unfolded_steps: int = 2):
    code = LinearCode(torch.eye(4, dtype=torch.float32))
    projections = make_toy_projection_set()
    llr = IdentityLLR()
    routing = CountingUniformRouting()
    decoder = Order2UnfoldedRPADecoder(
        code=code,
        projections=projections,
        bottom_decoder=HadamardOrder1Decoder(length=projections.n // 2),
        routing_policy=routing,
        num_unfolded_steps=num_unfolded_steps,
        channel_llr=llr,
    )
    return decoder, llr, routing


def test_resolve_unfolded_steps_uses_mode_depth_contract():
    full = uniform_full_mode()
    local = local_depth_mode(target_layer=1)
    clipped_local = local_depth_mode(target_layer=9)

    assert resolve_unfolded_steps(3, full) == 3
    assert resolve_unfolded_steps(3, local) == 2
    assert resolve_unfolded_steps(3, clipped_local) == 3


def test_forward_returns_logits_shape_for_rm_10_2_uniform_full():
    code = RMCode.from_text_file(LEGACY_G_TXT)
    projections = load_legacy_projection_file(LEGACY_PROJECTIONS_TXT, expected_count=512)
    decoder = Order2UnfoldedRPADecoder(
        code=code,
        projections=projections,
        bottom_decoder=HadamardOrder1Decoder(length=projections.n // 2),
        routing_policy=UniformRouting(),
        num_unfolded_steps=1,
        channel_llr=IdentityLLR(),
    )
    channel_output = torch.zeros(1, code.n)

    output = decoder(channel_output, uniform_full_mode())

    assert output.logits.shape == (1, code.n)
    assert output.stats["candidate_projections"] == 512
    assert output.stats["executed_projections"] == 512
    assert output.stats["aggregated_projections"] == 512


def test_decoder_passes_snr_context_to_awgn_llr():
    code = LinearCode(torch.eye(4, dtype=torch.float32))
    projections = make_toy_projection_set()
    decoder = Order2UnfoldedRPADecoder(
        code=code,
        projections=projections,
        bottom_decoder=HadamardOrder1Decoder(length=projections.n // 2),
        routing_policy=UniformRouting(),
        num_unfolded_steps=1,
        channel_llr=AWGNLLR(),
    )
    channel_output = torch.ones(1, code.n)
    mode = uniform_full_mode().with_channel_context(snr=2.0)

    output = decoder(channel_output, mode)

    assert output.logits.shape == (1, code.n)


def test_decoder_to_moves_code_and_projection_tensors_with_modules():
    decoder, _, _ = make_toy_decoder(num_unfolded_steps=1)

    moved = decoder.to(torch.device("cpu"))

    assert moved is decoder
    assert decoder.code.generator_matrix.device.type == "cpu"
    assert decoder.projections.coset_indices.device.type == "cpu"
    assert decoder.projections.flat_ids1.device.type == "cpu"
    assert decoder.bottom_decoder.hadamard_matrix.device.type == "cpu"


def test_decoder_to_cuda_moves_code_and_projection_tensors_when_available():
    if not torch.cuda.is_available():
        return
    decoder, _, _ = make_toy_decoder(num_unfolded_steps=1)

    decoder.to(torch.device("cuda"))

    assert decoder.code.generator_matrix.device.type == "cuda"
    assert decoder.projections.coset_indices.device.type == "cuda"
    assert decoder.projections.flat_ids1.device.type == "cuda"
    assert decoder.bottom_decoder.hadamard_matrix.device.type == "cuda"


def test_decoder_output_stats_contain_projection_counts():
    decoder, _, _ = make_toy_decoder(num_unfolded_steps=2)
    channel_output = torch.randn(2, decoder.code.n)

    output = decoder(channel_output, uniform_full_mode())

    assert output.stats["candidate_projections"] == 3
    assert output.stats["executed_projections"] == 3
    assert output.stats["aggregated_projections"] == 3
    assert output.stats["selection_scope"] == "full"
    assert output.stats["execution_mode"] == "compute_all_mask"
    assert output.stats["num_unfolded_steps_executed"] == 2
    assert len(output.stats["per_step_stats"]) == 2


def test_repeated_calls_do_not_depend_on_previous_calls_or_modes():
    torch.manual_seed(4)
    decoder, _, _ = make_toy_decoder(num_unfolded_steps=2)
    channel_output = torch.randn(2, decoder.code.n)
    full_mode = uniform_full_mode()
    local_mode = local_depth_mode(target_layer=0)

    first_full = decoder(channel_output, full_mode)
    local = decoder(channel_output, local_mode)
    second_full = decoder(channel_output, full_mode)

    torch.testing.assert_close(first_full.logits, second_full.logits)
    assert first_full.stats["num_unfolded_steps_executed"] == 2
    assert local.stats["num_unfolded_steps_executed"] == 1
    assert second_full.stats["num_unfolded_steps_executed"] == 2


def test_cpu_backward_smoke_through_uniform_full_decoder():
    torch.manual_seed(7)
    decoder, _, _ = make_toy_decoder(num_unfolded_steps=1)
    channel_output = torch.randn(2, decoder.code.n, requires_grad=True)

    output = decoder(channel_output, uniform_full_mode())
    output.logits.square().mean().backward()

    assert channel_output.grad is not None
    assert torch.isfinite(channel_output.grad).all()


def test_no_grad_frozen_prefix_steps_do_not_build_router_graph():
    torch.manual_seed(10)
    code = LinearCode(torch.eye(4, dtype=torch.float32))
    projections = make_toy_projection_set()
    router = StepwiseRouter([torch.nn.Linear(4, 3), torch.nn.Linear(4, 3)])
    decoder = Order2UnfoldedRPADecoder(
        code=code,
        projections=projections,
        bottom_decoder=HadamardOrder1Decoder(length=projections.n // 2),
        routing_policy=InputDependentRouterRouting(router),
        num_unfolded_steps=2,
        channel_llr=IdentityLLR(),
    )

    output = decoder(torch.randn(2, code.n), no_grad_frozen_mode(target_layer=1))

    assert output.stats["num_unfolded_steps_executed"] == 2
    assert output.stats["per_step_stats"][0]["ran_no_grad"] is True
    assert output.stats["per_step_stats"][1]["ran_no_grad"] is False
    first_logits = output.aux["steps"][0]["plan_aux"]["router_entropy_inputs"]
    second_logits = output.aux["steps"][1]["plan_aux"]["router_entropy_inputs"]
    assert first_logits.requires_grad is False
    assert second_logits.requires_grad is True
    assert output.logits.requires_grad is True


def test_uniform_full_decoder_uses_router_free_policy_only():
    decoder, llr, routing = make_toy_decoder(num_unfolded_steps=3)
    channel_output = torch.randn(2, decoder.code.n)

    decoder(channel_output, uniform_full_mode())

    assert list(decoder.routing_policy.parameters()) == []
    assert routing.plan_calls == 3
    assert llr.calls == 1


def test_decoder_has_no_hidden_training_phase_api():
    decoder, _, _ = make_toy_decoder()

    assert not hasattr(decoder, "set_training_phase")
    assert not hasattr(decoder, "active_training_layer")
