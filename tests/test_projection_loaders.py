from pathlib import Path

import pytest
import torch

from routing_rpa.projections.loaders import (
    load_legacy_projection_file,
    load_projection_artifact,
    load_projection_tensor_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PROJECTIONS_TXT = REPO_ROOT / "src_old" / "data" / "decoder" / "projections.txt"


def test_legacy_loader_returns_projection_set_with_expected_shape():
    projections = load_legacy_projection_file(LEGACY_PROJECTIONS_TXT, expected_count=512)

    assert projections.m == 10
    assert projections.n == 1024
    assert projections.subspace_dim == 1
    assert projections.num_projections == 512
    assert projections.directions.shape == (512,)
    assert projections.coset_indices.shape == (512, 2, 512)
    assert projections.flat_ids1.shape == (512 * 512,)
    assert projections.flat_ids2.shape == (512 * 512,)


def test_legacy_loader_metadata_contains_required_fields():
    projections = load_legacy_projection_file(LEGACY_PROJECTIONS_TXT, expected_count=512)

    for key in [
        "name",
        "m",
        "n",
        "subspace_dim",
        "num_projections",
        "family",
        "source",
        "notes",
    ]:
        assert key in projections.metadata

    assert projections.metadata["name"] == "rm10_half512"
    assert projections.metadata["num_projections"] == 512


def test_each_legacy_projection_covers_every_coordinate_once():
    projections = load_legacy_projection_file(LEGACY_PROJECTIONS_TXT, expected_count=512)

    rows = projections.coset_indices.reshape(projections.num_projections, projections.n)
    sorted_rows = torch.sort(rows, dim=1).values
    expected = torch.arange(projections.n, dtype=rows.dtype).expand_as(sorted_rows)

    assert torch.equal(sorted_rows, expected)


def test_legacy_loader_subset_preserves_shapes_and_flat_ids():
    projections = load_legacy_projection_file(LEGACY_PROJECTIONS_TXT, expected_count=512)

    subset = projections.subset(torch.tensor([0, 7, 31], dtype=torch.long))

    assert subset.num_projections == 3
    assert subset.coset_indices.shape == (3, 2, 512)
    assert subset.flat_ids1.shape == (3 * 512,)
    assert subset.flat_ids2.shape == (3 * 512,)
    assert torch.equal(subset.flat_ids1, subset.coset_indices[:, 0, :].reshape(-1))
    assert torch.equal(subset.flat_ids2, subset.coset_indices[:, 1, :].reshape(-1))


def test_legacy_loader_does_not_create_binary_cache_next_to_text():
    cache_path = LEGACY_PROJECTIONS_TXT.with_suffix(".pt")
    existed_before = cache_path.exists()

    load_legacy_projection_file(LEGACY_PROJECTIONS_TXT, expected_count=512)

    assert cache_path.exists() == existed_before


def test_projection_artifact_dispatch_loads_legacy_text():
    projections = load_projection_artifact(LEGACY_PROJECTIONS_TXT, expected_count=512)

    assert projections.num_projections == 512
    assert projections.coset_indices.shape == (512, 2, 512)


def test_projection_tensor_artifact_loader(tmp_path):
    artifact = tmp_path / "projections.pt"
    payload = {
        "m": 2,
        "n": 4,
        "subspace_dim": 1,
        "directions": torch.tensor([1, 2], dtype=torch.long),
        "coset_indices": torch.tensor(
            [
                [[0, 1], [2, 3]],
                [[0, 2], [1, 3]],
            ],
            dtype=torch.long,
        ),
        "metadata": {"name": "toy_tensor", "family": "unit_test"},
    }
    torch.save(payload, artifact)

    projections = load_projection_tensor_file(artifact, expected_count=2)

    assert projections.metadata["name"] == "toy_tensor"
    assert projections.metadata["source"] == "tensor_artifact"
    assert projections.metadata["m"] == 2
    assert projections.metadata["n"] == 4
    assert projections.metadata["subspace_dim"] == 1
    assert projections.metadata["num_projections"] == 2
    assert projections.num_projections == 2
    assert torch.equal(projections.directions, payload["directions"])


def test_projection_tensor_artifact_rejects_inconsistent_metadata(tmp_path):
    artifact = tmp_path / "bad_metadata.pt"
    torch.save(
        {
            "m": 2,
            "n": 4,
            "subspace_dim": 1,
            "coset_indices": torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long),
            "metadata": {
                "name": "bad_tensor",
                "m": 2,
                "n": 4,
                "subspace_dim": 1,
                "num_projections": 999,
            },
        },
        artifact,
    )

    with pytest.raises(ValueError, match="num_projections"):
        load_projection_tensor_file(artifact)


def test_projection_artifact_dispatch_loads_pt_tensor_artifact(tmp_path):
    artifact = tmp_path / "projections.pt"
    torch.save(
        {
            "m": 2,
            "n": 4,
            "subspace_dim": 1,
            "coset_indices": torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long),
            "metadata": {"name": "toy_tensor"},
        },
        artifact,
    )

    projections = load_projection_artifact(artifact, expected_count=1)

    assert projections.num_projections == 1
    assert projections.metadata["name"] == "toy_tensor"
