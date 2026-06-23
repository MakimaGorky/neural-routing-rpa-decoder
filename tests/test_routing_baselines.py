from unittest.mock import patch

import pytest
import torch

from routing_rpa.decoders.execution import ProjectionExecutor
from routing_rpa.decoders.modes import DecoderMode
from routing_rpa.decoders.routing import (
    RandomStaticRouting,
    SelectedStaticRouting,
    UniformRouting,
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


def uniform_full_mode() -> DecoderMode:
    return DecoderMode(
        selection_scope="full",
        execution_mode="compute_all_mask",
        top_k=None,
        forward_depth_policy="full_cascade",
        frozen_policy="frozen_weights",
    )


def random_static_mode(*, execution_mode: str = "compute_selected", top_k: int = 2) -> DecoderMode:
    return DecoderMode(
        selection_scope="static",
        execution_mode=execution_mode,
        top_k=top_k,
        forward_depth_policy="full_cascade",
        frozen_policy="frozen_weights",
    )


def selected_static_mode(*, top_k: int = 2) -> DecoderMode:
    return DecoderMode(
        selection_scope="static",
        execution_mode="compute_selected",
        top_k=top_k,
        forward_depth_policy="full_cascade",
        frozen_policy="frozen_weights",
    )


def test_uniform_routing_has_no_trainable_parameters():
    routing = UniformRouting()

    assert list(routing.parameters()) == []


def test_uniform_routing_plan_returns_equal_full_weights():
    state = torch.zeros(5, 4)
    projections = make_projection_set()

    plan = UniformRouting().plan(
        state=state,
        step=0,
        projections=projections,
        mode=uniform_full_mode(),
    )

    assert plan.selected_indices is None
    assert plan.candidate_count == 3
    assert plan.aggregation_weights.shape == (1, 3)
    assert plan.execution_weights.shape == (1, 3)
    torch.testing.assert_close(plan.aggregation_weights, torch.ones(1, 3))
    torch.testing.assert_close(plan.execution_weights, torch.ones(1, 3))


def test_uniform_full_reports_candidate_executed_aggregated_as_p_p_p():
    state = torch.zeros(2, 4)
    projections = make_projection_set()
    plan = UniformRouting().plan(state, 0, projections, uniform_full_mode())

    execution = ProjectionExecutor().resolve(projections, plan)

    assert execution.stats["candidate_projections"] == 3
    assert execution.stats["executed_projections"] == 3
    assert execution.stats["aggregated_projections"] == 3
    assert execution.stats["selection_scope"] == "full"
    assert execution.stats["execution_mode"] == "compute_all_mask"


def test_uniform_routing_rejects_non_uniform_full_mode():
    state = torch.zeros(1, 4)
    projections = make_projection_set()

    with pytest.raises(ValueError, match="selection_scope='full'"):
        UniformRouting().plan(
            state,
            0,
            projections,
            random_static_mode(execution_mode="compute_selected", top_k=2),
        )


def test_random_static_routing_has_no_trainable_parameters():
    routing = RandomStaticRouting(top_k=2, seed=123)

    assert list(routing.parameters()) == []


def test_random_static_subset_is_seed_reproducible():
    state = torch.zeros(1, 4)
    projections = make_projection_set()
    mode = random_static_mode(top_k=2)

    first = RandomStaticRouting(seed=19).plan(state, 0, projections, mode)
    second = RandomStaticRouting(seed=19).plan(state, 0, projections, mode)
    different = RandomStaticRouting(seed=7).plan(state, 0, projections, mode)

    assert torch.equal(first.selected_indices, second.selected_indices)
    assert not torch.equal(first.selected_indices, different.selected_indices)


def test_random_static_selected_subset_length_is_top_k():
    state = torch.zeros(1, 4)
    projections = make_projection_set()

    plan = RandomStaticRouting(seed=1).plan(
        state,
        step=0,
        projections=projections,
        mode=random_static_mode(top_k=2),
    )

    assert plan.selected_indices is not None
    assert plan.selected_indices.shape == (2,)
    assert plan.aggregation_weights.shape == (1, 2)
    assert plan.execution_weights.shape == (1, 2)


def test_random_static_compute_selected_reports_executed_k():
    state = torch.zeros(1, 4)
    projections = make_projection_set()
    plan = RandomStaticRouting(seed=3).plan(
        state,
        step=0,
        projections=projections,
        mode=random_static_mode(execution_mode="compute_selected", top_k=2),
    )

    execution = ProjectionExecutor().resolve(projections, plan)

    assert execution.projections.num_projections == 2
    assert execution.stats["candidate_projections"] == 3
    assert execution.stats["executed_projections"] == 2
    assert execution.stats["aggregated_projections"] == 2
    assert execution.projections is plan.aux["selected_projections"]


def test_random_static_compute_selected_caches_indices_and_projection_set():
    state = torch.zeros(1, 4)
    projections = make_projection_set()
    routing = RandomStaticRouting(seed=3)
    mode = random_static_mode(execution_mode="compute_selected", top_k=2)

    with patch("torch.randperm", wraps=torch.randperm) as randperm:
        first = routing.plan(state, step=0, projections=projections, mode=mode)
        second = routing.plan(state, step=1, projections=projections, mode=mode)

    assert randperm.call_count == 1
    assert first.selected_indices is second.selected_indices
    assert first.aux["selected_projections"] is second.aux["selected_projections"]


def test_random_static_diagnostic_compute_all_mask_reports_executed_p_aggregated_k():
    state = torch.zeros(1, 4)
    projections = make_projection_set()
    plan = RandomStaticRouting(seed=3).plan(
        state,
        step=0,
        projections=projections,
        mode=random_static_mode(execution_mode="compute_all_mask", top_k=2),
    )

    execution = ProjectionExecutor().resolve(projections, plan)

    assert execution.projections is projections
    assert execution.execution_weights.shape == (1, 3)
    assert execution.projection_weights.shape == (1, 3)
    assert execution.projection_weights.sum().item() == 2
    assert torch.count_nonzero(execution.projection_weights).item() == 2
    assert torch.all(execution.execution_weights == 1)
    assert execution.stats["candidate_projections"] == 3
    assert execution.stats["executed_projections"] == 3
    assert execution.stats["aggregated_projections"] == 2


def test_random_static_rejects_missing_or_out_of_range_top_k():
    state = torch.zeros(1, 4)
    projections = make_projection_set()

    with pytest.raises(ValueError, match="requires top_k"):
        RandomStaticRouting().plan(
            state,
            step=0,
            projections=projections,
            mode=DecoderMode(
                selection_scope="static",
                execution_mode="compute_selected",
                top_k=None,
                forward_depth_policy="full_cascade",
                frozen_policy="frozen_weights",
            ),
        )

    with pytest.raises(ValueError, match="top_k must be in"):
        RandomStaticRouting(seed=1).plan(
            state,
            step=0,
            projections=projections,
            mode=random_static_mode(top_k=4),
        )


def test_random_static_rejects_constructor_and_mode_top_k_mismatch():
    state = torch.zeros(1, 4)
    projections = make_projection_set()

    with pytest.raises(ValueError, match="top_k mismatch"):
        RandomStaticRouting(top_k=1, seed=0).plan(
            state,
            step=0,
            projections=projections,
            mode=random_static_mode(top_k=2),
        )


def test_selected_static_topk_reports_executed_k():
    state = torch.zeros(2, 4)
    projections = make_projection_set()
    routing = SelectedStaticRouting(torch.tensor([2, 0], dtype=torch.long))
    plan = routing.plan(
        state,
        step=0,
        projections=projections,
        mode=selected_static_mode(top_k=2),
    )

    execution = ProjectionExecutor().resolve(projections, plan)

    assert torch.equal(plan.selected_indices, torch.tensor([2, 0]))
    assert plan.aggregation_weights.shape == (1, 2)
    assert execution.projections.num_projections == 2
    assert execution.stats["candidate_projections"] == 3
    assert execution.stats["executed_projections"] == 2
    assert execution.stats["aggregated_projections"] == 2
    assert execution.stats["execution_mode"] == "compute_selected"
    assert execution.projections is plan.aux["selected_projections"]


def test_selected_static_topk_preserves_fixed_weights():
    state = torch.zeros(2, 4)
    projections = make_projection_set()
    routing = SelectedStaticRouting(
        torch.tensor([1, 2], dtype=torch.long),
        projection_weights=torch.tensor([0.25, 0.75]),
    )

    plan = routing.plan(
        state,
        step=0,
        projections=projections,
        mode=selected_static_mode(top_k=2),
    )

    torch.testing.assert_close(plan.aggregation_weights, torch.tensor([[0.25, 0.75]]))
    torch.testing.assert_close(plan.execution_weights, torch.tensor([[0.25, 0.75]]))


def test_selected_static_topk_rejects_compute_all_mask_to_avoid_fake_pruning():
    state = torch.zeros(1, 4)
    projections = make_projection_set()
    mode = DecoderMode(
        selection_scope="static",
        execution_mode="compute_all_mask",
        top_k=2,
        forward_depth_policy="full_cascade",
        frozen_policy="frozen_weights",
    )

    with pytest.raises(ValueError, match="compute_selected"):
        SelectedStaticRouting(torch.tensor([0, 2], dtype=torch.long)).plan(
            state,
            step=0,
            projections=projections,
            mode=mode,
        )


def test_selected_static_topk_caches_selected_projection_set():
    state = torch.zeros(2, 4)
    projections = make_projection_set()
    routing = SelectedStaticRouting(torch.tensor([2, 0], dtype=torch.long))

    first = routing.plan(
        state,
        step=0,
        projections=projections,
        mode=selected_static_mode(top_k=2),
    )
    second = routing.plan(
        state,
        step=1,
        projections=projections,
        mode=selected_static_mode(top_k=2),
    )

    assert first.aux["selected_projections"] is second.aux["selected_projections"]
