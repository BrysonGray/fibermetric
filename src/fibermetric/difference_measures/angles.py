"""Distances between periodic and antipodally symmetric directions."""

import numpy as np


def multiple_exclusive_distances(distances):
    """Greedily select pairwise distances without reusing rows or columns."""
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2 or 0 in distances.shape:
        raise ValueError('distances must be a nonempty 2D array.')

    count = min(distances.shape)
    available = distances.copy()
    selected = np.empty(count, dtype=float)
    for index in range(count):
        row, column = np.unravel_index(np.nanargmin(available), available.shape)
        selected[index] = distances[row, column]
        available[row, :] = np.nan
        available[:, column] = np.nan
    return selected


def periodic_distance_1d(first, second, period):
    """Return element-wise shortest distances between periodic angle arrays.

    Inputs contain scalar angles in their spatial dimensions and may optionally
    have a singleton final angle axis. The output has the broadcasted spatial
    shape.
    """
    if period <= 0:
        raise ValueError('period must be positive.')
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim and first.shape[-1] == 1:
        first = first[..., 0]
    if second.ndim and second.shape[-1] == 1:
        second = second[..., 0]
    try:
        difference = np.abs(first - second) % period
    except ValueError as error:
        raise ValueError('first and second must have broadcastable spatial dimensions.') from error
    return np.minimum(difference, period - difference)


def apsym_vector_distance(first, second):
    """Return element-wise antipodally symmetric distances between vector arrays.

    Inputs have shape ``(..., D)`` with broadcastable leading dimensions. The
    output has the broadcasted leading shape.
    """
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim < 1 or second.ndim < 1 or first.shape[-1] != second.shape[-1]:
        raise ValueError('first and second must have equal vector dimensions on their final axes.')

    first_norms = np.linalg.norm(first, axis=-1, keepdims=True)
    second_norms = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(first_norms == 0) or np.any(second_norms == 0):
        raise ValueError('direction vectors must be nonzero.')
    cosine = np.sum(
        (first / first_norms) * (second / second_norms),
        axis=-1,
    )
    return np.arccos(np.clip(np.abs(cosine), 0, 1))