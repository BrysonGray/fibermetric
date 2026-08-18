"""Orientation tensor and direction routines."""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize


def structure_tensor(image, derivative_sigma=1.0, tensor_sigma=1.0, normalize=True, masked=False, id_minus_S=False):
    """Construct 2D or 3D structure tensors from a grayscale image."""
    if image.dtype == np.uint8:
        image = image.astype(float) / 255

    if image.ndim == 2:
        grad_x = gaussian_filter(image, sigma=[derivative_sigma, derivative_sigma], order=(0, 1))
        grad_y = gaussian_filter(image, sigma=[derivative_sigma, derivative_sigma], order=(1, 0))
        norm = np.sqrt(grad_x ** 2 + grad_y ** 2) + np.finfo(float).eps
        if normalize:
            grad_x = grad_x / norm
            grad_y = grad_y / norm
        xx = gaussian_filter(grad_x * grad_x, sigma=[tensor_sigma, tensor_sigma])
        xy = gaussian_filter(grad_x * grad_y, sigma=[tensor_sigma, tensor_sigma])
        yy = gaussian_filter(grad_y * grad_y, sigma=[tensor_sigma, tensor_sigma])
        tensors = np.stack((1 - xx, -xy, -xy, 1 - yy), axis=-1)
        if masked:
            tensors[norm < 1e-9] = None
        return tensors.reshape(tensors.shape[:-1] + (2, 2))

    if image.ndim == 3:
        grad_x = gaussian_filter(image, sigma=[derivative_sigma, derivative_sigma, derivative_sigma], order=(0, 0, 1))
        grad_y = gaussian_filter(image, sigma=[derivative_sigma, derivative_sigma, derivative_sigma], order=(0, 1, 0))
        grad_z = gaussian_filter(image, sigma=[derivative_sigma, derivative_sigma, derivative_sigma], order=(1, 0, 0))
        norm = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2) + np.finfo(float).eps
        if normalize:
            grad_x = grad_x / norm
            grad_y = grad_y / norm
            grad_z = grad_z / norm
        xx = gaussian_filter(grad_x * grad_x, sigma=[tensor_sigma, tensor_sigma, tensor_sigma])
        yy = gaussian_filter(grad_y * grad_y, sigma=[tensor_sigma, tensor_sigma, tensor_sigma])
        zz = gaussian_filter(grad_z * grad_z, sigma=[tensor_sigma, tensor_sigma, tensor_sigma])
        xy = gaussian_filter(grad_x * grad_y, sigma=[tensor_sigma, tensor_sigma, tensor_sigma])
        xz = gaussian_filter(grad_x * grad_z, sigma=[tensor_sigma, tensor_sigma, tensor_sigma])
        yz = gaussian_filter(grad_y * grad_z, sigma=[tensor_sigma, tensor_sigma, tensor_sigma])
        tensors = np.stack((xx, xy, xz, xy, yy, yz, xz, yz, zz), axis=-1)
        tensors = tensors.reshape(tensors.shape[:-1] + (3, 3))
        if not id_minus_S:
            return -tensors
        return np.eye(3) - tensors

    raise Exception(f'Input must be a 2 or 3 dimensional array but found: {image.ndim}')


def anisotropy(eigenvalues):
    """Calculate anisotropy from 2D or 3D eigenvalues."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim < 1 or eigenvalues.shape[-1] not in (2, 3):
        count = eigenvalues.shape[-1] if eigenvalues.ndim else 0
        raise ValueError(f'Accepts 2 or 3 eigenvalues but found {count}')

    count = eigenvalues.shape[-1]
    if count == 2:
        numerator = np.abs(eigenvalues[..., 0] - eigenvalues[..., 1])
        denominator = np.abs(eigenvalues[..., 0] + eigenvalues[..., 1])
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > np.finfo(float).eps,
        )

    mean = np.mean(eigenvalues, axis=-1, keepdims=True)
    numerator = np.sum((eigenvalues - mean) ** 2, axis=-1)
    denominator = np.sum(eigenvalues ** 2, axis=-1)
    values = np.sqrt(
        np.divide(
            1.5 * numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > np.finfo(float).eps,
        )
    )
    maximum = np.max(values, initial=0.0)
    return values / maximum if maximum > np.finfo(float).eps else values


def angles(tensors, cartesian=False):
    """Compute principal directions from structure tensors.
    Parameters
    ----------
    tensors : array_like
        Structure tensors of shape (..., 2, 2) or (..., 3, 3).
    cartesian : bool, optional
        If True, return the principal direction vectors in Cartesian coordinates.
        If False, return the angles (theta, phi) in radians. Default is False.
    Returns
    -------
    array_like
        Principal directions of shape (M, N) for 2D tensors or (L, M, N, 2) for 3D tensors,
        where the last dimension contains the angles (theta, phi) in radians if `cartesian` is False,
        or the Cartesian coordinates of the principal direction vectors if `cartesian` is True.

    """
    eigenvalues, eigenvectors = np.linalg.eigh(tensors)
    vectors = eigenvectors[..., -1]
    if cartesian:
        return vectors
    if eigenvalues.shape[-1] == 2:
        return np.arctan(vectors[..., 0] / (vectors[..., 1] + np.finfo(float).eps))
    x_coord = vectors[..., 0]
    y_coord = vectors[..., 1]
    z_coord = vectors[..., 2]
    theta = np.arctan(np.sqrt(x_coord ** 2 + y_coord ** 2) / (z_coord + np.finfo(float).eps))
    theta = np.where(theta < 0, theta + np.pi, theta)
    phi = np.arctan(x_coord / (y_coord + np.finfo(float).eps))
    return np.stack((theta, phi), axis=-1)


def hsv(tensors, image):
    """Compute orientation, anisotropy, and an RGB tensor visualization."""
    import matplotlib

    tensors = np.asarray(tensors, dtype=float)
    image = np.asarray(image, dtype=float)
    if tensors.shape[-2:] != (2, 2):
        raise ValueError('tensors must end with shape (2, 2).')
    if image.ndim != tensors.ndim - 2:
        raise ValueError('image rank must match the tensor spatial rank.')
    eigenvalues, eigenvectors = np.linalg.eigh(tensors)
    vectors = eigenvectors[..., -1]
    theta = np.mod(np.arctan2(vectors[..., 1], vectors[..., 0]), np.pi) / np.pi
    anisotropy_values = anisotropy(eigenvalues)
    if tensors.shape[:-2] != image.shape:
        image = resize(image, tensors.shape[:-2], anti_aliasing=True)
    stack = np.stack([theta, anisotropy_values, image], -1)
    return theta, anisotropy_values, matplotlib.colors.hsv_to_rgb(stack)