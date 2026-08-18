"""Orientation direction helpers and clustering."""

import numpy as np


def project_to_plane(vectors, normal, transform=None):
    """Project 3D vectors onto a plane through the origin."""
    vectors = np.asarray(vectors, dtype=float)
    normal = np.asarray(normal, dtype=float)
    if vectors.ndim < 1 or normal.ndim < 1 or vectors.shape[-1] != normal.shape[-1]:
        raise ValueError('vectors and normal must have equal vector dimensions.')
    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    if np.any(normal_norm <= np.finfo(float).eps):
        raise ValueError('normal vectors must be nonzero.')
    normal = normal / normal_norm
    projection = np.sum(vectors * normal, axis=-1, keepdims=True)
    projection = projection * normal
    projected = vectors - projection
    if transform is not None:
        transform = np.asarray(transform, dtype=float)
        if transform.shape[-2:] != (vectors.shape[-1], vectors.shape[-1]):
            raise ValueError('transform must end with square vector dimensions.')
        projected = np.einsum('...ij,...j->...i', transform, projected)
    return projected


def angle_to_rgb(angle, brightness=1):
    """Map an angle to an RGB triple."""
    angle = np.asarray(angle)
    brightness = np.asarray(brightness)
    red = np.abs(brightness * np.sin(angle))
    green = np.abs(brightness * np.sin(angle + 2 * np.pi / 3.0))
    blue = np.abs(brightness * np.sin(angle + 4 * np.pi / 3.0))
    return np.stack(np.broadcast_arrays(red, green, blue), axis=-1)


def vec_to_theta(vector):
    """Convert 2D vectors to polar angle."""
    vector = np.asarray(vector)
    if vector.ndim < 1 or vector.shape[-1] != 2:
        raise ValueError('vector must have 2 components on its final axis.')
    return np.arctan2(vector[..., 0], vector[..., 1])


def circular_odf(
    angles,
    shape_out=None,
    ntheta=500,
    n_coeffs=None,
    decay=0.1,
    normalize=True,
):
    """Convert 2D principal directions into truncated Fourier ODF coefficients.

    The angular samples are mirrored by pi to enforce antipodal symmetry,
    histogrammed over [-pi, pi], transformed with rFFT, then truncated.
    """
    from ..auxiliary.utils import gather

    angles = np.asarray(angles, dtype=float)
    if angles.ndim == 0:
        angles = angles[None]
    if shape_out is None:
        samples = angles.reshape((1, -1))
        output_shape = ()
    else:
        shape_out = tuple(shape_out)
        gathered = gather(angles, shape_out)
        samples = gathered.reshape((-1, gathered.shape[-1]))
        output_shape = shape_out

    available_coefficients = ntheta // 2 + 1
    if n_coeffs is None:
        n_coeffs = available_coefficients
    if n_coeffs < 1:
        raise ValueError('n_coeffs must be at least 1.')
    n_coeffs = min(n_coeffs, available_coefficients)

    theta = np.arange(ntheta + 1) * (2 * np.pi / ntheta) - np.pi
    coefficients = np.empty((len(samples), n_coeffs), dtype=complex)
    for index, values in enumerate(samples):
        values = (values + np.pi / 2) % np.pi - np.pi / 2
        mirrored = np.where(values < 0, values + np.pi, values - np.pi)
        histogram, _ = np.histogram(np.concatenate((values, mirrored)), theta)
        transformed = np.fft.rfft(histogram)
        if decay is not None:
            transformed *= np.exp(-decay * np.arange(len(transformed)))
        transformed = transformed[:n_coeffs]
        if normalize and np.abs(transformed[0]) > np.finfo(float).eps:
            transformed = transformed / transformed[0]
        coefficients[index] = transformed

    return coefficients.reshape(output_shape + (n_coeffs,))


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
    if coefficients.ndim < 1:
        raise ValueError('coefficients must be at least a 1D array.')

    theta = np.arange(ntheta) * (2 * np.pi / ntheta) - np.pi
    histogram = np.fft.irfft(coefficients, ntheta, axis=-1)

    if nonnegative:
        histogram = np.clip(histogram, 0.0, None)

    if normalize:
        total = np.sum(histogram, axis=-1, keepdims=True)
        histogram = np.divide(
            histogram,
            total,
            out=np.zeros_like(histogram),
            where=np.abs(total) > np.finfo(float).eps,
        )

    return histogram, theta


def circular_odf_directions(
    coefficients,
    max_directions=3,
    relative_threshold=0.1,
    ntheta=500,
    chunk_size=1024,
):
    """Extract principal axes in bounded histogram chunks using finite differences.

    Returns angles in ``[-pi / 2, pi / 2)`` with shape
    ``(..., max_directions)``. Missing directions are represented by NaN.
    """
    if max_directions < 1:
        raise ValueError('max_directions must be at least 1.')
    if not 0 <= relative_threshold <= 1:
        raise ValueError('relative_threshold must be between 0 and 1.')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive.')

    coefficients = np.asarray(coefficients)
    if coefficients.ndim < 1:
        raise ValueError('coefficients must be at least a 1D array.')
    spatial_shape = coefficients.shape[:-1]
    flat_coefficients = coefficients.reshape((-1, coefficients.shape[-1]))
    directions = np.full((len(flat_coefficients), max_directions), np.nan)

    for start in range(0, len(flat_coefficients), chunk_size):
        stop = min(start + chunk_size, len(flat_coefficients))
        histograms, theta = circular_odf_to_histogram(
            flat_coefficients[start:stop],
            ntheta=ntheta,
            normalize=False,
            nonnegative=True,
        )
        for index, values in enumerate(histograms, start=start):
            backward_difference = values - np.roll(values, 1)
            forward_difference = np.roll(values, -1) - values
            peak_indices = np.flatnonzero(
                (backward_difference > 0) & (forward_difference <= 0)
            )
            if not len(peak_indices):
                continue

            cutoff = relative_threshold * values[peak_indices].max()
            peak_indices = peak_indices[values[peak_indices] >= cutoff]
            peak_indices = peak_indices[np.argsort(values[peak_indices])[::-1]]

            selected = []
            for peak_index in peak_indices:
                angle = (theta[peak_index] + np.pi / 2) % np.pi - np.pi / 2
                if all(abs(np.angle(np.exp(2j * (angle - other)))) > 2 * np.pi / ntheta for other in selected):
                    selected.append(angle)
                if len(selected) == max_directions:
                    break
            directions[index, :len(selected)] = selected

    return directions.reshape(spatial_shape + (max_directions,))


def spherical_odf(
    directions,
    coordinates='cartesian',
    shape_out=None,
    sh_order_max=8,
    sphere=None,
    normalize=True,
):
    """Convert collections of 3D directions into symmetric SH ODF coefficients.

    Parameters
    ----------
    directions : array_like
        Spatial direction array with shape ``(..., 3)`` for Cartesian vectors
        or ``(..., 2)`` for spherical angles.
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
    from ..auxiliary.utils import gather

    directions = np.asarray(directions, dtype=float)
    if directions.ndim < 1 or directions.size == 0:
        raise ValueError('directions must contain at least one direction.')
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

    if shape_out is None:
        samples = directions.reshape((1, -1, 3))
        output_shape = ()
    else:
        shape_out = tuple(shape_out)
        samples = gather(directions, shape_out, feature_axis=-1)
        samples = samples.reshape((-1, samples.shape[-2], 3))
        output_shape = shape_out

    if sphere is None:
        sphere = get_sphere(name='symmetric362')
    vertices = np.asarray(sphere.vertices)
    histogram = np.zeros((len(samples), len(vertices)), dtype=float)
    for index, vectors in enumerate(samples):
        bins = np.argmax(np.concatenate((vectors, -vectors)) @ vertices.T, axis=-1)
        np.add.at(histogram[index], bins, 1)

    if normalize:
        histogram /= histogram.sum(axis=-1, keepdims=True)

    _, inverse_basis = sh_to_sf_matrix(
        sphere,
        sh_order_max=sh_order_max,
        basis_type='descoteaux07',
        return_inv=True,
        legacy=False,
    )
    coefficients = histogram @ inverse_basis
    return coefficients.reshape(output_shape + (coefficients.shape[-1],))


def _spherical_odf_basis(sphere, sh_order_max):
    from dipy.reconst.shm import sh_to_sf_matrix

    basis, _ = sh_to_sf_matrix(
        sphere,
        sh_order_max=sh_order_max,
        basis_type='descoteaux07',
        legacy=False,
    )
    return basis


def _spherical_coefficients_to_histogram(
    coefficients,
    basis,
    normalize=True,
    nonnegative=True,
):
    histogram = np.asarray(coefficients) @ basis
    if nonnegative:
        histogram = np.clip(histogram, 0.0, None)
    if normalize:
        total = np.sum(histogram, axis=-1, keepdims=True)
        histogram = np.divide(
            histogram,
            total,
            out=np.zeros_like(histogram),
            where=total > np.finfo(float).eps,
        )
    return histogram


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

    coefficients = np.asarray(coefficients)
    if coefficients.ndim < 1:
        raise ValueError('coefficients must be at least a 1D array.')
    if sphere is None:
        sphere = get_sphere(name='repulsion724')

    basis = _spherical_odf_basis(sphere, sh_order_max)
    histogram = _spherical_coefficients_to_histogram(
        coefficients,
        basis,
        normalize=normalize,
        nonnegative=nonnegative,
    )
    return histogram, sphere


def spherical_odf_directions(
    coefficients,
    max_directions=3,
    relative_threshold=0.1,
    sphere=None,
    sh_order_max=8,
    chunk_size=1024,
):
    """Extract principal axes in bounded histogram chunks using finite differences.

    Returns Cartesian unit vectors with shape ``(..., max_directions, 3)``.
    Missing directions are represented by NaN.
    """
    from dipy.data import get_sphere

    if max_directions < 1:
        raise ValueError('max_directions must be at least 1.')
    if not 0 <= relative_threshold <= 1:
        raise ValueError('relative_threshold must be between 0 and 1.')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive.')
    if sphere is None:
        sphere = get_sphere(name='symmetric362')

    coefficients = np.asarray(coefficients)
    if coefficients.ndim < 1:
        raise ValueError('coefficients must be at least a 1D array.')
    spatial_shape = coefficients.shape[:-1]
    flat_coefficients = coefficients.reshape((-1, coefficients.shape[-1]))
    vertices = np.asarray(sphere.vertices)
    edges = np.asarray(sphere.edges)
    basis = _spherical_odf_basis(sphere, sh_order_max)
    directions = np.full((len(flat_coefficients), max_directions, 3), np.nan)

    for start in range(0, len(flat_coefficients), chunk_size):
        stop = min(start + chunk_size, len(flat_coefficients))
        histograms = _spherical_coefficients_to_histogram(
            flat_coefficients[start:stop],
            basis,
            normalize=False,
            nonnegative=True,
        )
        for index, values in enumerate(histograms, start=start):
            neighbor_maximum = np.full(len(vertices), -np.inf)
            neighbor_minimum = np.full(len(vertices), np.inf)
            np.maximum.at(neighbor_maximum, edges[:, 0], values[edges[:, 1]])
            np.maximum.at(neighbor_maximum, edges[:, 1], values[edges[:, 0]])
            np.minimum.at(neighbor_minimum, edges[:, 0], values[edges[:, 1]])
            np.minimum.at(neighbor_minimum, edges[:, 1], values[edges[:, 0]])
            peak_indices = np.flatnonzero(
                (values >= neighbor_maximum) & (values > neighbor_minimum)
            )
            if not len(peak_indices):
                continue

            cutoff = relative_threshold * values[peak_indices].max()
            peak_indices = peak_indices[values[peak_indices] >= cutoff]
            peak_indices = peak_indices[np.argsort(values[peak_indices])[::-1]]

            selected = []
            for peak_index in peak_indices:
                vector = vertices[peak_index]
                if all(abs(np.dot(vector, other)) < 1 - np.finfo(float).eps for other in selected):
                    selected.append(vector)
                if len(selected) == max_directions:
                    break
            directions[index, :len(selected)] = selected

    return directions.reshape(spatial_shape + (max_directions, 3))