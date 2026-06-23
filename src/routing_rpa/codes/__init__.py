"""Binary code domain objects."""

from routing_rpa.codes.gf2 import (
    gf2_matmul,
    gf2_rank,
    gf2_row_reduce,
    has_full_row_rank,
    is_binary_tensor,
    require_full_row_rank,
    validate_binary_tensor,
)
from routing_rpa.codes.linear import LinearCode
from routing_rpa.codes.reed_muller import RMCode, rm_dimension

__all__ = [
    "LinearCode",
    "RMCode",
    "gf2_matmul",
    "gf2_rank",
    "gf2_row_reduce",
    "has_full_row_rank",
    "is_binary_tensor",
    "require_full_row_rank",
    "rm_dimension",
    "validate_binary_tensor",
]
