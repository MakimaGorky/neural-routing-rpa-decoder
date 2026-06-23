import pytest
import torch

from routing_rpa.decoders.execution import ProjectionExecutor
from routing_rpa.decoders.selection import ProjectionPlan
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


def test_compute_all_mask_keeps_all_projections_and_reports_stats():
    projections = make_projection_set()
    plan = ProjectionPlan(
        candidate_count=projections.num_projections,
        selected_indices=None,
        selection_scope="full",
        execution_mode="compute_all_mask",
        aggregation_weights=torch.ones(1, projections.num_projections),
        execution_weights=torch.ones(1, projections.num_projections),
    )

    execution = ProjectionExecutor().resolve(projections, plan)

    assert execution.projections is projections
    assert execution.execution_weights.shape == (1, 3)
    assert execution.projection_weights.shape == (1, 3)
    assert execution.aggregation_weights.shape == (1, 3)
    assert execution.stats == {
        "candidate_projections": 3,
        "executed_projections": 3,
        "aggregated_projections": 3,
        "selection_scope": "full",
        "execution_mode": "compute_all_mask",
    }


def test_compute_all_mask_can_report_masked_aggregated_count_from_plan_aux():
    projections = make_projection_set()
    plan = ProjectionPlan(
        candidate_count=projections.num_projections,
        selected_indices=torch.tensor([0, 2], dtype=torch.long),
        selection_scope="static",
        execution_mode="compute_all_mask",
        aggregation_weights=torch.tensor([[1.0, 0.0, 1.0]]),
        execution_weights=torch.ones(1, projections.num_projections),
        aux={"aggregated_count": 2},
    )

    execution = ProjectionExecutor().resolve(projections, plan)

    assert execution.projections is projections
    assert execution.stats["candidate_projections"] == 3
    assert execution.stats["executed_projections"] == 3
    assert execution.stats["aggregated_projections"] == 2
    torch.testing.assert_close(
        execution.projection_weights,
        torch.tensor([[1.0, 0.0, 1.0]]),
    )
    torch.testing.assert_close(execution.execution_weights, torch.ones(1, 3))


def test_compute_all_mask_rejects_aggregation_weights_length_not_p():
    projections = make_projection_set()
    plan = ProjectionPlan(
        candidate_count=projections.num_projections,
        selected_indices=torch.tensor([0, 2], dtype=torch.long),
        selection_scope="static",
        execution_mode="compute_all_mask",
        aggregation_weights=torch.ones(1, 2),
        execution_weights=torch.ones(1, projections.num_projections),
    )

    with pytest.raises(ValueError, match="aggregation_weights"):
        ProjectionExecutor().resolve(projections, plan)


def test_compute_selected_static_narrows_projection_set():
    projections = make_projection_set()
    selected_indices = torch.tensor([2, 0], dtype=torch.long)
    plan = ProjectionPlan(
        candidate_count=projections.num_projections,
        selected_indices=selected_indices,
        selection_scope="static",
        execution_mode="compute_selected",
        aggregation_weights=torch.ones(1, 2),
        execution_weights=torch.ones(1, 2),
    )

    execution = ProjectionExecutor().resolve(projections, plan)

    assert execution.projections is not projections
    assert execution.projections.num_projections == 2
    assert torch.equal(execution.projections.directions, torch.tensor([3, 1]))
    assert execution.execution_weights.shape == (1, 2)
    assert execution.stats == {
        "candidate_projections": 3,
        "executed_projections": 2,
        "aggregated_projections": 2,
        "selection_scope": "static",
        "execution_mode": "compute_selected",
    }


def test_compute_selected_really_returns_k_projection_tensors():
    projections = make_projection_set()
    selected_indices = torch.tensor([2, 0], dtype=torch.long)
    plan = ProjectionPlan(
        candidate_count=projections.num_projections,
        selected_indices=selected_indices,
        selection_scope="static",
        execution_mode="compute_selected",
        aggregation_weights=torch.ones(1, 2),
        execution_weights=torch.ones(1, 2),
    )

    execution = ProjectionExecutor().resolve(projections, plan)

    assert execution.projections.num_projections == 2
    assert execution.projections.coset_indices.shape == (2, 2, projections.n // 2)
    assert execution.projections.flat_ids1.shape == (2 * (projections.n // 2),)
    assert execution.projections.flat_ids2.shape == (2 * (projections.n // 2),)


def test_compute_selected_requires_selected_indices():
    projections = make_projection_set()
    plan = ProjectionPlan(
        candidate_count=projections.num_projections,
        selected_indices=None,
        selection_scope="static",
        execution_mode="compute_selected",
        aggregation_weights=torch.ones(1, 2),
        execution_weights=torch.ones(1, 2),
    )

    with pytest.raises(ValueError, match="selected_indices"):
        ProjectionExecutor().resolve(projections, plan)


def test_per_sample_compute_selected_is_explicitly_not_implemented():
    projections = make_projection_set()
    plan = ProjectionPlan(
        candidate_count=projections.num_projections,
        selected_indices=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        selection_scope="per_sample",
        execution_mode="compute_selected",
        aggregation_weights=torch.ones(2, 2),
        execution_weights=torch.ones(2, 2),
    )

    with pytest.raises(NotImplementedError, match="per_sample compute_selected"):
        ProjectionExecutor().resolve(projections, plan)


def test_executor_rejects_candidate_count_mismatch():
    projections = make_projection_set()
    plan = ProjectionPlan(
        candidate_count=99,
        selected_indices=None,
        selection_scope="full",
        execution_mode="compute_all_mask",
        aggregation_weights=torch.ones(1, 3),
        execution_weights=torch.ones(1, 3),
    )

    with pytest.raises(ValueError, match="candidate_count"):
        ProjectionExecutor().resolve(projections, plan)
