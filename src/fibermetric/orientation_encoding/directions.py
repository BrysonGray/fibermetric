"""Orientation direction helpers and clustering."""

import numpy as np


def project_to_plane(vectors, normal, transform=None):
    """Project 3D vectors onto a plane through the origin."""
    vectors = np.asarray(vectors)
    normal = np.asarray(normal)
    normal = normal / np.sum(normal ** 2) ** 0.5
    projection = np.einsum('...i,i->...', vectors, normal)
    projection = projection[..., None] * normal[None]
    projected = vectors - projection
    if transform is not None:
        transform = np.asarray(transform)
        projected = np.einsum('ij,...j->...i', transform, projected)
    return projected


def angle_to_rgb(angle, brightness=1):
    """Map an angle to an RGB triple."""
    red = np.abs(brightness * np.sin(angle))
    green = np.abs(brightness * np.sin(angle + 2 * np.pi / 3.0))
    blue = np.abs(brightness * np.sin(angle + 4 * np.pi / 3.0))
    return [red, green, blue]


def vec_to_theta(vector):
    """Convert 2D vectors to polar angle."""
    return np.arctan(vector[..., 0] / (vector[..., 1] + np.finfo(float).eps))


def circular_odf(angles, ntheta=500, n_coeffs=None, decay=0.1, normalize=True):
    """Convert 2D principal directions into truncated Fourier ODF coefficients.

    The angular samples are mirrored by pi to enforce antipodal symmetry,
    histogrammed over [-pi, pi], transformed with rFFT, then truncated.
    """
    angles = np.asarray(angles)
    angles_flat = angles.reshape(-1)
    mirrored = np.where(angles_flat < 0, angles_flat + np.pi, angles_flat - np.pi)
    angles_sym = np.concatenate((angles_flat, mirrored), axis=0)

    theta = np.arange(ntheta + 1) * (2 * np.pi / ntheta) - np.pi
    histogram, _ = np.histogram(angles_sym, theta)
    coefficients = np.fft.rfft(histogram)

    if decay is not None:
        index = np.arange(len(coefficients))
        coefficients = coefficients * np.exp(-decay * index)

    if n_coeffs is None:
        n_coeffs = len(coefficients)
    if n_coeffs < 1:
        raise ValueError('n_coeffs must be at least 1.')

    n_coeffs = min(n_coeffs, len(coefficients))
    truncated = coefficients[:n_coeffs]

    if normalize and np.abs(truncated[0]) > np.finfo(float).eps:
        truncated = truncated / truncated[0]

    return truncated


def circular_odf_to_histogram(coefficients, ntheta=500, normalize=True, nonnegative=True):
    """Reconstruct a polar histogram from truncated 2D ODF Fourier coefficients.
    Parameters
    ----------
    coefficients : array_like
        Fourier coefficients of shape (..., n_coeffs).
    ntheta : int, optional
        Number of angular bins for the histogram. Default is 500.
    normalize : bool, optional
        If True, normalize the histogram to sum to 1. Default is True.
    nonnegative : bool, optional
        If True, clip negative values in the histogram to zero. Default is True.
    Returns
    -------
    histogram : ndarray
        Reconstructed polar histogram of shape (..., ntheta).
    theta : ndarray
        Array of angular bin centers of shape (ntheta,)."""
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 1:
        raise ValueError('coefficients must be a 1D array.')

    theta = np.arange(ntheta + 1) * (2 * np.pi / ntheta) - np.pi
    histogram = np.fft.irfft(coefficients, ntheta + 1)

    if nonnegative:
        histogram = np.clip(histogram, 0.0, None)

    if normalize:
        total = np.sum(histogram)
        if np.abs(total) > np.finfo(float).eps:
            histogram = histogram / total

    return histogram, theta


def spherical_odf(directions, coordinates='cartesian', sh_order_max=8, sphere=None, normalize=True):
    """Convert collections of 3D directions into symmetric SH ODF coefficients.

    Parameters
    ----------
    directions : array_like
        Array of shape (..., N, 3) or (..., N, 2)
        containing N 3D Cartesian or 2D spherical directions.
    coordinates : {'cartesian', 'spherical'}, optional
        Specifies the coordinate system of the input directions.
        'cartesian' expects 3D vectors (x, y, z).
        'spherical' expects 2D angles (theta, phi) in radians.
    sh_order_max : int, optional
        Maximum SH order for the output coefficients. Must be a nonnegative even integer.
    sphere : Sphere, optional
        Sphere object defining the vertices for the ODF. If None, uses the 'symmetric362' sphere from DIPY.
    normalize : bool, optional
        If True, normalize the ODF coefficients to have a maximum of 1.
        
    Returns
    -------
    coefficients : ndarray
        Array of shape (..., n_coeffs) containing the SH coefficients of the ODF.
    """
    from dipy.data import get_sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    directions = np.asarray(directions, dtype=float)
    if directions.ndim < 2 or directions.shape[-2] == 0:
        raise ValueError('directions must contain at least one collection of directions.')
    if sh_order_max < 0 or sh_order_max % 2:
        raise ValueError('sh_order_max must be a nonnegative even integer.')
    if coordinates == 'spherical':
        if directions.shape[-1] != 2:
            raise ValueError('Spherical directions must contain polar and azimuth angles.')
        polar = directions[..., 0]
        azimuth = directions[..., 1]
        directions = np.stack((
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
            np.cos(polar),
        ), axis=-1)
    elif coordinates != 'cartesian':
        raise ValueError("coordinates must be 'cartesian' or 'spherical'.")
    elif directions.shape[-1] != 3:
        raise ValueError('Cartesian directions must contain x, y, and z components.')

    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError('Directions must be nonzero vectors.')
    directions = directions / norms

    if sphere is None:
        sphere = get_sphere(name='symmetric362')
    vertices = np.asarray(sphere.vertices)
    flat = directions.reshape((-1, directions.shape[-2], 3))
    histogram = np.zeros((len(flat), len(vertices)), dtype=float)
    for index, vectors in enumerate(flat):
        bins = np.argmax(np.concatenate((vectors, -vectors)) @ vertices.T, axis=-1)
        np.add.at(histogram[index], bins, 1)

    if normalize:
        histogram /= histogram.sum(axis=-1, keepdims=True)

    _, inverse_basis = sh_to_sf_matrix(sphere, sh_order_max=sh_order_max, basis_type='descoteaux07', return_inv=True)
    coefficients = histogram @ inverse_basis
    return coefficients.reshape(directions.shape[:-2] + (coefficients.shape[-1],))


def spherical_odf_to_histogram(coefficients, sphere=None, sh_order_max=8, normalize=True, nonnegative=True):
    """Reconstruct a polar histogram from truncated 3D ODF SH coefficients.
    
    Parameters
    ----------
    coefficients : array_like
        Spherical harmonic coefficients of shape (..., n_coeffs).
    sphere : Sphere, optional
        Sphere object defining the vertices for the histogram. If None, uses the 'symmetric362' sphere from DIPY.
    normalize : bool, optional
        If True, normalize the histogram to sum to 1.
    nonnegative : bool, optional
        If True, clip negative values in the histogram to zero.
    """
    from dipy.data import get_sphere
    from dipy.reconst.shm import sh_to_sf

    coefficients = np.asarray(coefficients)
    if coefficients.ndim < 1:
        raise ValueError('coefficients must be at least a 1D array.')
    if sphere is None:
        sphere = get_sphere(name='repulsion724')

    histogram = sh_to_sf(coefficients, sphere, sh_order_max=sh_order_max, basis_type='descoteaux07')

    if nonnegative:
        histogram = np.clip(histogram, 0.0, None)

    if normalize:
        total = np.sum(histogram, axis=-1, keepdims=True)
        histogram = np.divide(histogram, total, out=np.zeros_like(histogram), where=total > np.finfo(float).eps)


    return histogram, sphere