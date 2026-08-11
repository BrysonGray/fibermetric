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
    if eigenvalues.shape[-1] == 3:
        eigenvalues = eigenvalues.transpose(3, 0, 1, 2)
        trace = np.sum(eigenvalues, axis=0)
        anisotropy_values = np.sqrt((3 / 2) * (np.sum((eigenvalues - (1 / 3) * trace) ** 2, axis=0) / np.sum(eigenvalues ** 2, axis=0)))
        anisotropy_values = np.nan_to_num(anisotropy_values)
        return anisotropy_values / np.max(anisotropy_values)
    if eigenvalues.shape[-1] == 2:
        return abs(eigenvalues[..., 0] - eigenvalues[..., 1]) / abs(eigenvalues[..., 0] + eigenvalues[..., 1])
    raise Exception(f'Accepts 2 or 3 eigenvalues but found {eigenvalues.shape[-1]}')


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
    """Compute orientation, anisotropy, and an RGB visualization from 2D tensors."""
    import matplotlib

    if image.ndim != 2:
        raise Exception(f'Only accepts two dimensional images but found {image.ndim} dimensions')
    eigenvalues, eigenvectors = np.linalg.eigh(tensors)
    vectors = eigenvectors[..., -1]
    theta = ((np.arctan(vectors[..., 1] / vectors[..., 0])) + np.pi / 2) / np.pi
    anisotropy_values = anisotropy(eigenvalues)
    if tensors.shape[:-2] != image.shape:
        down = [x // y for x, y in zip(image.shape, tensors.shape[:-2])]
        image = resize(image, (image.shape[0] // down[0], image.shape[1] // down[1]), anti_aliasing=True)
    stack = np.stack([theta, anisotropy_values, image], -1)
    return theta, anisotropy_values, matplotlib.colors.hsv_to_rgb(stack)