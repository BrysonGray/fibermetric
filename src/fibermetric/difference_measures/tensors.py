"""Distances between symmetric positive-definite tensor images."""

import numpy as np


def _validate_spd_pair(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim < 2 or second.ndim < 2:
        raise ValueError('Tensor images must be at least two-dimensional.')
    if first.shape[-2:] not in ((2, 2), (3, 3)) or second.shape[-2:] != first.shape[-2:]:
        raise ValueError('Tensor images must end with shape (2, 2) or (3, 3).')
    try:
        spatial_shape = np.broadcast_shapes(first.shape[:-2], second.shape[:-2])
    except ValueError as error:
        raise ValueError('Tensor images must have broadcastable spatial dimensions.') from error
    first = np.broadcast_to(first, spatial_shape + first.shape[-2:])
    second = np.broadcast_to(second, spatial_shape + second.shape[-2:])
    if not np.allclose(first, np.swapaxes(first, -1, -2)):
        raise ValueError('The first tensor image must be symmetric.')
    if not np.allclose(second, np.swapaxes(second, -1, -2)):
        raise ValueError('The second tensor image must be symmetric.')

    first_eigenvalues, first_eigenvectors = np.linalg.eigh(first)
    second_eigenvalues = np.linalg.eigvalsh(second)
    if np.any(first_eigenvalues <= 0) or np.any(second_eigenvalues <= 0):
        raise ValueError('Tensor images must be positive-definite.')
    return first, second, first_eigenvalues, first_eigenvectors


def riemannian_tensor_distance(first, second):
    """Return the affine-invariant Riemannian distance between SPD tensors."""
    first, second, eigenvalues, eigenvectors = _validate_spd_pair(first, second)
    inverse_sqrt = (
        eigenvectors * (1 / np.sqrt(eigenvalues))[..., None, :]
    ) @ np.swapaxes(eigenvectors, -1, -2)
    relative = inverse_sqrt @ second @ inverse_sqrt
    relative_eigenvalues = np.linalg.eigvalsh(relative)
    return np.sqrt(np.sum(np.log(relative_eigenvalues) ** 2, axis=-1))


def symmetric_kl_tensor_distance(first, second):
    """Return the averaged symmetric KL divergence between Gaussian tensors."""
    first, second, _, _ = _validate_spd_pair(first, second)
    dimensions = first.shape[-1]
    first_to_second = np.trace(np.linalg.solve(second, first), axis1=-2, axis2=-1)
    second_to_first = np.trace(np.linalg.solve(first, second), axis1=-2, axis2=-1)
    return 0.25 * (first_to_second + second_to_first - 2 * dimensions)


def tensor_distance(first, second, metric='riemannian'):
    """Compare SPD tensor images with the selected metric."""
    if metric == 'riemannian':
        return riemannian_tensor_distance(first, second)
    if metric == 'symmetric_kl':
        return symmetric_kl_tensor_distance(first, second)
    raise ValueError("metric must be 'riemannian' or 'symmetric_kl'.")