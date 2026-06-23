"""Hot tensor kernels for the order-2 unfolded RPA decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from routing_rpa.projections.projection_set import ProjectionSet


class _StraightThroughSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output.clamp(-1, 1)


def _differentiable_sign(input: Tensor) -> Tensor:
    return _StraightThroughSign.apply(input)


def _require_same_device(*, tensor: Tensor, projections: ProjectionSet, name: str) -> None:
    if projections.coset_indices.device != tensor.device:
        raise ValueError(
            f"{name} and projections must be on the same device: "
            f"{tensor.device} != {projections.coset_indices.device}"
        )


def convert_to_llr(received: Tensor, sigma_sqr: float) -> Tensor:
    """Convert BPSK AWGN channel output to LLRs using the legacy formula."""
    if sigma_sqr <= 0:
        raise ValueError(f"sigma_sqr must be positive, got {sigma_sqr}")
    return 2.0 * received / sigma_sqr


def project_1d(llr: Tensor, projections: ProjectionSet) -> Tensor:
    """Project length-n LLRs onto one-dimensional coset pairs.

    Args:
        llr: Tensor with shape ``(B, n)``.
        projections: One-dimensional projection geometry with shape
            ``(P, 2, n / 2)``.

    Returns:
        Tensor with shape ``(B, P, n / 2)``.
    """
    if llr.dim() != 2:
        raise ValueError(f"llr must have shape (B, n), got {llr.shape}")
    if llr.shape[1] != projections.n:
        raise ValueError(
            f"llr length must match projections.n={projections.n}, got {llr.shape[1]}"
        )
    _require_same_device(tensor=llr, projections=projections, name="llr")

    ids1 = projections.coset_indices[:, 0, :]
    ids2 = projections.coset_indices[:, 1, :]
    first = llr[:, ids1]
    second = llr[:, ids2]
    return F.softplus(first + second) - torch.logaddexp(first, second)


def hadamard_decode_order1(projected: Tensor, H: Tensor) -> Tensor:
    """Dense Hadamard bottom decoder for projected first-order RM words.

    Args:
        projected: Tensor with shape ``(B, P_or_K, n / 2)``.
        H: Dense Hadamard matrix with shape ``(n / 2, n / 2)``.

    Returns:
        Tensor with the same shape as ``projected``.
    """
    if projected.dim() != 3:
        raise ValueError(f"projected must have shape (B, P, L), got {projected.shape}")
    if H.dim() != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be a square matrix, got {H.shape}")
    if projected.shape[-1] != H.shape[0]:
        raise ValueError(
            f"projected last dimension must match H size, got {projected.shape[-1]} "
            f"and {H.shape[0]}"
        )
    if projected.device != H.device:
        raise ValueError(
            f"projected and H must be on the same device: {projected.device} != {H.device}"
        )
    if projected.dtype != H.dtype:
        raise ValueError(f"projected and H must have the same dtype: {projected.dtype} != {H.dtype}")

    transformed = projected @ H
    max_indices = torch.argmax(torch.abs(transformed), dim=-1, keepdim=True)
    mask = torch.zeros_like(transformed)
    mask.scatter_(dim=-1, index=max_indices, value=1.0)
    max_components = transformed * mask
    return (max_components @ H) / H.shape[0]


def aggregate_1d(
    received_llr: Tensor,
    decoded_projected: Tensor,
    projection_weights: Tensor,
    projections: ProjectionSet,
    *,
    check_norm: bool = False,
) -> Tensor:
    """Aggregate projected decoded words back to length-n LLRs.

    ``projection_weights`` accepts either shared weights ``(1, P_or_K)`` or
    per-batch weights ``(B, P_or_K)``. The function intentionally does not add
    epsilon or clamp the normalization term; optional checks can be enabled via
    ``check_norm=True``.
    """
    if received_llr.dim() != 2:
        raise ValueError(f"received_llr must have shape (B, n), got {received_llr.shape}")
    if decoded_projected.dim() != 3:
        raise ValueError(
            f"decoded_projected must have shape (B, P, n/2), got {decoded_projected.shape}"
        )
    if projection_weights.dim() != 2:
        raise ValueError(
            "projection_weights must have shape (1, P) or (B, P), "
            f"got {projection_weights.shape}"
        )
    if received_llr.shape[0] != decoded_projected.shape[0]:
        raise ValueError(
            "received_llr and decoded_projected batch sizes must match: "
            f"{received_llr.shape[0]} != {decoded_projected.shape[0]}"
        )
    if received_llr.shape[1] != projections.n:
        raise ValueError(
            f"received_llr length must match projections.n={projections.n}, "
            f"got {received_llr.shape[1]}"
        )
    batch_size = received_llr.shape[0]
    projection_count = projections.num_projections
    half_n = projections.n // 2
    if decoded_projected.shape[1:] != (projection_count, half_n):
        raise ValueError(
            "decoded_projected must have shape "
            f"(B, {projection_count}, {half_n}), got {decoded_projected.shape}"
        )
    if projection_weights.shape[0] not in {1, batch_size}:
        raise ValueError(
            "projection_weights first dimension must be 1 or batch size "
            f"{batch_size}, got {projection_weights.shape[0]}"
        )
    if projection_weights.shape[1] != projection_count:
        raise ValueError(
            f"projection_weights length must match projection count {projection_count}, "
            f"got {projection_weights.shape[1]}"
        )
    _require_same_device(tensor=received_llr, projections=projections, name="received_llr")
    if decoded_projected.device != received_llr.device:
        raise ValueError(
            "decoded_projected and received_llr must be on the same device: "
            f"{decoded_projected.device} != {received_llr.device}"
        )
    if projection_weights.device != received_llr.device:
        raise ValueError(
            "projection_weights and received_llr must be on the same device: "
            f"{projection_weights.device} != {received_llr.device}"
        )

    weights = projection_weights.to(dtype=received_llr.dtype)
    weights_expanded = weights.unsqueeze(-1)
    decoded_sign = _differentiable_sign(decoded_projected)

    ids1 = projections.coset_indices[:, 0, :]
    ids2 = projections.coset_indices[:, 1, :]
    vals1 = received_llr[:, ids1]
    vals2 = received_llr[:, ids2]

    weighted_sign = decoded_sign * weights_expanded
    term1 = vals2 * weighted_sign
    term2 = vals1 * weighted_sign

    flat_ids1 = projections.flat_ids1.unsqueeze(0).expand(batch_size, -1)
    flat_ids2 = projections.flat_ids2.unsqueeze(0).expand(batch_size, -1)
    flat_weights = weights_expanded.expand(batch_size, -1, half_n).reshape(batch_size, -1)

    word = torch.zeros_like(received_llr)
    word.scatter_add_(1, flat_ids1, term1.reshape(batch_size, -1))
    word.scatter_add_(1, flat_ids2, term2.reshape(batch_size, -1))

    norm = torch.zeros_like(received_llr)
    norm.scatter_add_(1, flat_ids1, flat_weights)
    norm.scatter_add_(1, flat_ids2, flat_weights)

    if check_norm and torch.any(norm <= 0):
        raise ValueError("aggregate_1d normalization must be positive")

    return word / norm


__all__ = [
    "aggregate_1d",
    "convert_to_llr",
    "hadamard_decode_order1",
    "project_1d",
]
