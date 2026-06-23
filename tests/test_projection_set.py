import pytest
import torch

from routing_rpa.projections.projection_set import ProjectionSet


def make_toy_projection_set() -> ProjectionSet:
    return ProjectionSet.from_coset_indices(
        m=2,
        n=4,
        subspace_dim=1,
        directions=torch.tensor([1, 2], dtype=torch.long),
        coset_indices=torch.tensor(
            [
                [[0, 1], [2, 3]],
                [[0, 2], [1, 3]],
            ],
            dtype=torch.long,
        ),
        metadata={"name": "toy", "num_projections": 2},
    )


def test_projection_set_exposes_count_and_flat_ids():
    projections = make_toy_projection_set()

    assert projections.num_projections == 2
    assert projections.coset_indices.shape == (2, 2, 2)
    assert torch.equal(projections.flat_ids1, torch.tensor([0, 1, 0, 2]))
    assert torch.equal(projections.flat_ids2, torch.tensor([2, 3, 1, 3]))


def test_projection_set_subset_preserves_valid_shapes_and_flat_ids():
    projections = make_toy_projection_set()

    subset = projections.subset(torch.tensor([1], dtype=torch.long))

    assert subset.num_projections == 1
    assert subset.coset_indices.shape == (1, 2, 2)
    assert torch.equal(subset.directions, torch.tensor([2]))
    assert torch.equal(subset.flat_ids1, torch.tensor([0, 2]))
    assert torch.equal(subset.flat_ids2, torch.tensor([1, 3]))
    assert subset.metadata["name"] == "toy"
    assert subset.metadata["parent_num_projections"] == 2
    assert subset.metadata["num_projections"] == 1


def test_projection_set_subset_rejects_duplicate_indices():
    projections = make_toy_projection_set()

    with pytest.raises(ValueError, match="duplicate"):
        projections.subset(torch.tensor([1, 1], dtype=torch.long))


def test_projection_set_subset_uses_trusted_constructor_without_full_validation(monkeypatch):
    projections = make_toy_projection_set()

    def fail_validation(**kwargs):
        raise AssertionError("full validation should not run for trusted subsets")

    monkeypatch.setattr(
        "routing_rpa.projections.projection_set.validate_projection_fields",
        fail_validation,
    )

    subset = projections.subset(torch.tensor([0], dtype=torch.long))

    assert subset.num_projections == 1
    assert torch.equal(subset.flat_ids1, subset.coset_indices[:, 0, :].reshape(-1))


def test_projection_set_to_moves_tensors_and_keeps_metadata():
    projections = make_toy_projection_set()

    moved = projections.to(torch.device("cpu"))

    assert moved.coset_indices.device.type == "cpu"
    assert moved.directions.device.type == "cpu"
    assert moved.flat_ids1.device.type == "cpu"
    assert moved.metadata == projections.metadata
    assert moved.metadata is not projections.metadata


def test_invalid_projection_set_fails_validation():
    with pytest.raises(ValueError, match="cover every coordinate"):
        ProjectionSet.from_coset_indices(
            m=2,
            n=4,
            subspace_dim=1,
            directions=torch.tensor([1], dtype=torch.long),
            coset_indices=torch.tensor([[[0, 1], [1, 3]]], dtype=torch.long),
        )


def test_projection_set_validates_flat_id_cache():
    cosets = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)

    with pytest.raises(ValueError, match="flat_ids1"):
        ProjectionSet(
            m=2,
            n=4,
            subspace_dim=1,
            directions=torch.tensor([1], dtype=torch.long),
            coset_indices=cosets,
            metadata={},
            flat_ids1=torch.tensor([0, 0], dtype=torch.long),
            flat_ids2=cosets[:, 1, :].reshape(-1),
        )
