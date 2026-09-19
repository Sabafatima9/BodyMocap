"""Length-proportional rotation distribution along bone chains (pure Python).

For a chain of N bones with rest lengths L_i that must realise a total
relative rotation R_total = (axis a, angle theta), bone i receives the local
increment

    theta_i = theta * (L_i / sum(L))     about the same axis a,

so the cumulative (armature-space) rotation at the end of bone i is
``R_total ** c_i`` with ``c_i = sum_{j<=i} L_j / sum(L)`` (arc-length
parameterisation).  Because every increment shares the axis, the product of
the increments equals R_total exactly, i.e. the chain end reaches the target
orientation while bending/twisting smoothly -- no single joint kinks.

The solver uses these helpers for spine/neck bend-twist distribution and for
forearm/shin twist distribution; ``remap_chain`` offers the same distribution
for N:M transfer of per-bone local rotations.
"""

from __future__ import annotations

from typing import List, Sequence

from ..core.math3d import (
    cumulative_quat_product,
    cumulative_weights,
    normalize_weights,
    quat_conjugate,
    quat_identity,
    quat_normalize,
    quat_pow,
)
from ..core.types import Quat


def chain_parameter_breaks(rest_lengths: Sequence[float]) -> List[float]:
    """Normalised arc-length parameter at each joint: [0, c_1, ..., 1]."""
    if not rest_lengths:
        return [0.0, 1.0]
    return [0.0] + cumulative_weights(rest_lengths)


def distribute_rotation(total: Quat, lengths: Sequence[float]) -> List[Quat]:
    """Local increments theta_i = theta * L_i / sum(L) about the total axis."""
    return [quat_pow(total, w) for w in normalize_weights(lengths)]


def cumulative_rotations(total: Quat, lengths: Sequence[float]) -> List[Quat]:
    """Armature-space rotation reached at the end of each bone (R ** c_i)."""
    return [quat_pow(total, c) for c in cumulative_weights(lengths)]


def remap_chain(
    source_deltas: Sequence[Quat],
    source_lengths: Sequence[float],
    target_lengths: Sequence[float],
) -> List[Quat]:
    """N:M remap of chain-local rotations preserving the chain-end rotation.

    The source chain's total rotation (product of its local rotations) is
    redistributed over the target bones proportionally to their lengths.  The
    product of the returned rotations equals the source product (up to the
    single-axis approximation of the total), for any N and M.
    """
    m = len(target_lengths)
    if m == 0:
        return []
    if not source_deltas:
        return [quat_identity() for _ in range(m)]
    if len(source_deltas) == m and list(source_lengths) == list(target_lengths):
        return [quat_normalize(q) for q in source_deltas]
    total = quat_normalize(cumulative_quat_product(source_deltas))
    return distribute_rotation(total, target_lengths)


def chain_product(quats: Sequence[Quat]) -> Quat:
    return quat_normalize(cumulative_quat_product(quats))


def relative_rotation(a: Quat, b: Quat) -> Quat:
    """Rotation r with r @ a == b."""
    return quat_normalize(b @ quat_conjugate(a))
