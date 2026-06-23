import torch

from routing_rpa.codes.linear import LinearCode
from routing_rpa.decoders.bottom_decoders import HadamardOrder1Decoder
from routing_rpa.decoders.kernels_order2 import aggregate_1d, project_1d
from routing_rpa.decoders.modes import DecoderMode
from routing_rpa.decoders.order2_unfolded import Order2UnfoldedRPADecoder
from routing_rpa.decoders.routing import RoutingPolicy, SelectedStaticRouting
from routing_rpa.decoders.selection import ProjectionPlan
from routing_rpa.projections.projection_set import ProjectionSet


class IdentityLLR:
    def __call__(self, received: torch.Tensor, code: LinearCode) -> torch.Tensor:
        return received


class StaticMaskedAllRouting(RoutingPolicy):
    def __init__(self, selected_indices: torch.Tensor, selected_weights: torch.Tensor) -> None:
        super().__init__()
        self.selected_indices = selected_indices.to(dtype=torch.long)
        self.selected_weights = selected_weights

    def plan(self, state, step, projections, mode):
        full_weights = torch.zeros(
            1,
            projections.num_projections,
            device=state.device,
            dtype=state.dtype,
        )
        full_weights.scatter_(
            1,
            self.selected_indices.to(device=state.device).unsqueeze(0),
            self.selected_weights.to(device=state.device, dtype=state.dtype).unsqueeze(0),
        )
        return ProjectionPlan(
            candidate_count=projections.num_projections,
            selected_indices=self.selected_indices.to(device=state.device),
            selection_scope="static",
            execution_mode="compute_all_mask",
            aggregation_weights=full_weights,
            execution_weights=torch.ones_like(full_weights),
            aux={"aggregated_count": int(self.selected_indices.numel())},
        )


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


def selected_mode() -> DecoderMode:
    return DecoderMode(
        selection_scope="static",
        execution_mode="compute_selected",
        top_k=2,
        forward_depth_policy="full_cascade",
        frozen_policy="frozen_weights",
    )


def masked_mode() -> DecoderMode:
    return DecoderMode(
        selection_scope="static",
        execution_mode="compute_all_mask",
        top_k=2,
        forward_depth_policy="full_cascade",
        frozen_policy="frozen_weights",
    )


def test_compute_selected_kernel_output_matches_compute_all_mask_for_fixed_subset():
    torch.manual_seed(1)
    projections = make_projection_set()
    selected_indices = torch.tensor([2, 0], dtype=torch.long)
    selected_weights = torch.tensor([[0.75, 1.25]], dtype=torch.float32)
    full_weights = torch.tensor([[1.25, 0.0, 0.75]], dtype=torch.float32)
    received = torch.randn(3, projections.n)

    full_projected = project_1d(received, projections)
    selected_projections = projections.subset(selected_indices)
    selected_projected = project_1d(received, selected_projections)

    torch.testing.assert_close(
        selected_projected,
        full_projected.index_select(1, selected_indices),
    )

    bottom_decoder = HadamardOrder1Decoder(length=projections.n // 2)
    full_decoded = bottom_decoder(full_projected, projections, step=0)
    selected_decoded = bottom_decoder(selected_projected, selected_projections, step=0)
    full_output = aggregate_1d(
        received_llr=received,
        decoded_projected=full_decoded,
        projection_weights=full_weights,
        projections=projections,
    )
    selected_output = aggregate_1d(
        received_llr=received,
        decoded_projected=selected_decoded,
        projection_weights=selected_weights,
        projections=selected_projections,
    )

    torch.testing.assert_close(selected_output, full_output)


def test_compute_selected_decoder_output_matches_compute_all_mask_for_fixed_subset():
    torch.manual_seed(2)
    code = LinearCode(torch.eye(4, dtype=torch.float32))
    projections = make_projection_set()
    selected_indices = torch.tensor([2, 0], dtype=torch.long)
    selected_weights = torch.tensor([0.75, 1.25], dtype=torch.float32)
    channel_output = torch.randn(2, code.n)

    selected_decoder = Order2UnfoldedRPADecoder(
        code=code,
        projections=projections,
        bottom_decoder=HadamardOrder1Decoder(length=projections.n // 2),
        routing_policy=SelectedStaticRouting(
            selected_indices,
            projection_weights=selected_weights,
        ),
        num_unfolded_steps=1,
        channel_llr=IdentityLLR(),
    )
    masked_decoder = Order2UnfoldedRPADecoder(
        code=code,
        projections=projections,
        bottom_decoder=HadamardOrder1Decoder(length=projections.n // 2),
        routing_policy=StaticMaskedAllRouting(selected_indices, selected_weights),
        num_unfolded_steps=1,
        channel_llr=IdentityLLR(),
    )

    selected_output = selected_decoder(channel_output, selected_mode())
    masked_output = masked_decoder(channel_output, masked_mode())

    torch.testing.assert_close(selected_output.logits, masked_output.logits)
    assert selected_output.stats["executed_projections"] == 2
    assert masked_output.stats["executed_projections"] == 3
