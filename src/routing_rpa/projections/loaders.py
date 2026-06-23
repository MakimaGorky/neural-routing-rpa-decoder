"""Projection artifact loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from math import log2

import torch

from routing_rpa.projections.projection_set import ProjectionSet


def _parse_int_line(line: str, *, path: Path, line_number: int) -> list[int]:
    try:
        return [int(value) for value in line.split()]
    except ValueError as exc:
        raise ValueError(f"Invalid integer value in {path} at line {line_number}") from exc


def _require_metadata_match(
    metadata: dict[str, Any],
    key: str,
    expected: int,
    *,
    path: Path,
) -> None:
    if key not in metadata:
        return
    actual = int(metadata[key])
    if actual != expected:
        raise ValueError(
            f"Projection tensor artifact metadata field {key!r} disagrees with "
            f"loaded tensors in {path}: metadata={actual}, expected={expected}"
        )


def load_legacy_projection_file(
    path: str | Path,
    *,
    m: int = 10,
    n: int | None = None,
    device: torch.device | str = "cpu",
    expected_count: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProjectionSet:
    """Load the legacy three-line projection text format.

    Each record consists of one direction/difference line followed by two
    coset-half index lines.
    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Projection artifact not found: {artifact_path}")

    target_n = 2**m if n is None else n
    half_n = target_n // 2
    target_device = torch.device(device)

    directions: list[int] = []
    cosets: list[list[list[int]]] = []
    nonempty_lines: list[tuple[int, str]] = []
    with artifact_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if stripped:
                nonempty_lines.append((line_number, stripped))

    if len(nonempty_lines) % 3 != 0:
        raise ValueError(
            f"Legacy projection file must contain triples of non-empty lines, "
            f"got {len(nonempty_lines)} lines"
        )

    for offset in range(0, len(nonempty_lines), 3):
        direction_line_number, direction_line = nonempty_lines[offset]
        first_line_number, first_line = nonempty_lines[offset + 1]
        second_line_number, second_line = nonempty_lines[offset + 2]

        direction_values = _parse_int_line(
            direction_line,
            path=artifact_path,
            line_number=direction_line_number,
        )
        if len(direction_values) != 1:
            raise ValueError(
                f"Projection direction at line {direction_line_number} must be a single integer"
            )

        first_half = _parse_int_line(first_line, path=artifact_path, line_number=first_line_number)
        second_half = _parse_int_line(
            second_line,
            path=artifact_path,
            line_number=second_line_number,
        )
        if len(first_half) != half_n or len(second_half) != half_n:
            raise ValueError(
                f"Projection halves must each have length {half_n}; "
                f"got {len(first_half)} and {len(second_half)}"
            )

        directions.append(direction_values[0])
        cosets.append([first_half, second_half])

    if expected_count is not None and len(directions) != expected_count:
        raise ValueError(
            f"Expected {expected_count} projections in {artifact_path}, got {len(directions)}"
        )

    projection_count = len(directions)
    default_metadata: dict[str, Any] = {
        "name": "rm10_half512" if m == 10 and projection_count == 512 else f"rm{m}_legacy{projection_count}",
        "m": m,
        "n": target_n,
        "subspace_dim": 1,
        "num_projections": projection_count,
        "family": "one_dimensional_voting_set",
        "source": "sage",
        "notes": "legacy text projection set",
    }
    if metadata:
        default_metadata.update(metadata)

    return ProjectionSet.from_coset_indices(
        m=m,
        n=target_n,
        subspace_dim=1,
        directions=torch.tensor(directions, dtype=torch.long, device=target_device),
        coset_indices=torch.tensor(cosets, dtype=torch.long, device=target_device),
        metadata=default_metadata,
    )


def load_projection_tensor_file(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    expected_count: int | None = None,
) -> ProjectionSet:
    """Load a tensor projection artifact saved as a dictionary in a .pt file."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Projection artifact not found: {artifact_path}")

    payload = torch.load(
        artifact_path,
        map_location=torch.device(device),
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError(
            f"Projection tensor artifact must be a dict, got {type(payload)!r}"
        )
    if "coset_indices" not in payload:
        raise ValueError("Projection tensor artifact missing 'coset_indices'")

    metadata = dict(payload.get("metadata", {}))
    coset_indices = payload["coset_indices"]
    if not isinstance(coset_indices, torch.Tensor):
        raise TypeError("'coset_indices' must be a torch.Tensor")
    if coset_indices.dim() != 3:
        raise ValueError(
            f"'coset_indices' must have shape (P, 2, n/2), got {coset_indices.shape}"
        )

    projection_count = int(coset_indices.shape[0])
    inferred_n = int(coset_indices.shape[2] * 2)
    if inferred_n <= 0 or inferred_n & (inferred_n - 1):
        raise ValueError(f"Cannot infer projection length n from shape {coset_indices.shape}")
    inferred_m = int(log2(inferred_n))

    directions = payload.get("directions")
    if directions is None:
        directions = torch.arange(projection_count, dtype=torch.long, device=coset_indices.device)
    if not isinstance(directions, torch.Tensor):
        raise TypeError("'directions' must be a torch.Tensor when present")

    raw_m = payload.get("m", metadata.get("m"))
    raw_n = payload.get("n", metadata.get("n"))
    m = inferred_m if raw_m is None else int(raw_m)
    n = inferred_n if raw_n is None else int(raw_n)
    if m != inferred_m:
        raise ValueError(
            f"Projection tensor artifact m={m} disagrees with inferred m={inferred_m}"
        )
    if n != inferred_n:
        raise ValueError(
            f"Projection tensor artifact n={n} disagrees with inferred n={inferred_n}"
        )
    subspace_dim = int(payload.get("subspace_dim", metadata.get("subspace_dim", 1)))

    _require_metadata_match(metadata, "m", m, path=artifact_path)
    _require_metadata_match(metadata, "n", n, path=artifact_path)
    _require_metadata_match(metadata, "subspace_dim", subspace_dim, path=artifact_path)
    _require_metadata_match(metadata, "num_projections", projection_count, path=artifact_path)

    metadata["m"] = m
    metadata["n"] = n
    metadata["subspace_dim"] = subspace_dim
    metadata["num_projections"] = projection_count
    metadata.setdefault("source", "tensor_artifact")

    if expected_count is not None and projection_count != expected_count:
        raise ValueError(
            f"Expected {expected_count} projections in {artifact_path}, got {projection_count}"
        )

    target_device = torch.device(device)
    return ProjectionSet.from_coset_indices(
        m=m,
        n=n,
        subspace_dim=subspace_dim,
        directions=directions.to(device=target_device),
        coset_indices=coset_indices.to(device=target_device),
        metadata=metadata,
    )


def load_projection_artifact(
    path: str | Path,
    *,
    m: int = 10,
    n: int | None = None,
    device: torch.device | str = "cpu",
    expected_count: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProjectionSet:
    """Load a ProjectionSet from a supported projection artifact path."""
    artifact_path = Path(path)
    suffix = artifact_path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        if metadata is not None:
            raise ValueError("metadata override is only supported for legacy text artifacts")
        return load_projection_tensor_file(
            artifact_path,
            device=device,
            expected_count=expected_count,
        )

    return load_legacy_projection_file(
        artifact_path,
        m=m,
        n=n,
        device=device,
        expected_count=expected_count,
        metadata=metadata,
    )


__all__ = [
    "load_legacy_projection_file",
    "load_projection_artifact",
    "load_projection_tensor_file",
]
