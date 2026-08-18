"""Distances between circular and spherical ODF images."""

import numpy as np
import ot

from ..orientation_encoding.directions import circular_odf_to_histogram
from ..orientation_encoding.directions import _spherical_coefficients_to_histogram
from ..orientation_encoding.directions import _spherical_odf_basis


def _normalize_histograms(histograms):
    histograms = np.asarray(histograms, dtype=float)
    totals = np.sum(histograms, axis=-1, keepdims=True)
    if np.any(totals <= np.finfo(float).eps):
        raise ValueError('ODFs must have positive mass.')
    return histograms / totals


def _histogram_distance(first, second, metric, ground_cost=None):
    first = _normalize_histograms(first)
    second = _normalize_histograms(second)
    if first.shape[-1] != second.shape[-1]:
        raise ValueError('ODF histograms must have the same number of bins.')
    try:
        spatial_shape = np.broadcast_shapes(first.shape[:-1], second.shape[:-1])
    except ValueError as error:
        raise ValueError('ODF images must have broadcastable spatial dimensions.') from error
    first = np.broadcast_to(first, spatial_shape + first.shape[-1:])
    second = np.broadcast_to(second, spatial_shape + second.shape[-1:])
    if metric == 'total_variation':
        return 0.5 * np.sum(np.abs(first - second), axis=-1)
    if metric != 'wasserstein':
        raise ValueError("metric must be 'total_variation' or 'wasserstein'.")

    flat_first = first.reshape((-1, first.shape[-1]))
    flat_second = second.reshape((-1, second.shape[-1]))
    distances = np.array([
        ot.emd2(source, target, ground_cost)
        for source, target in zip(flat_first, flat_second)
    ])
    return distances.reshape(first.shape[:-1])


def _chunked_coefficient_distance(
    first,
    second,
    histogram_converter,
    metric,
    ground_cost,
    chunk_size,
):
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive.')
    if metric not in ('total_variation', 'wasserstein'):
        raise ValueError("metric must be 'total_variation' or 'wasserstein'.")

    first = np.asarray(first)
    second = np.asarray(second)
    if first.ndim < 1 or second.ndim < 1:
        raise ValueError('ODF coefficients must be at least 1D arrays.')
    if first.shape[-1] != second.shape[-1]:
        raise ValueError('ODFs must have the same number of coefficients.')
    try:
        spatial_shape = np.broadcast_shapes(first.shape[:-1], second.shape[:-1])
    except ValueError as error:
        raise ValueError('ODF images must have broadcastable spatial dimensions.') from error

    coefficient_count = first.shape[-1]
    first = np.broadcast_to(first, spatial_shape + (coefficient_count,))
    second = np.broadcast_to(second, spatial_shape + (coefficient_count,))
    distances = np.empty(int(np.prod(spatial_shape, dtype=int)) or 1, dtype=float)
    iterator = np.nditer(
        (first, second),
        flags=('external_loop', 'buffered', 'zerosize_ok'),
        op_flags=(('readonly',), ('readonly',)),
        order='C',
        buffersize=chunk_size * coefficient_count,
    )

    start = 0
    for first_chunk, second_chunk in iterator:
        first_chunk = first_chunk.reshape((-1, coefficient_count))
        second_chunk = second_chunk.reshape((-1, coefficient_count))
        first_histogram = histogram_converter(first_chunk)
        second_histogram = histogram_converter(second_chunk)
        stop = start + len(first_chunk)
        distances[start:stop] = _histogram_distance(
            first_histogram,
            second_histogram,
            metric,
            ground_cost,
        )
        start = stop
    return distances[:start].reshape(spatial_shape)


def circular_odf_distance(
    first,
    second,
    metric='total_variation',
    ntheta=500,
    chunk_size=1024,
):
    """Compare circular ODF images using bounded histogram chunks."""
    theta = np.arange(ntheta) * (2 * np.pi / ntheta) - np.pi
    ground_cost = None
    if metric == 'wasserstein':
        difference = theta[:, None] - theta[None, :]
        ground_cost = 0.5 * np.abs(np.angle(np.exp(2j * difference)))

    def histogram_converter(coefficients):
        return circular_odf_to_histogram(
            coefficients,
            ntheta=ntheta,
            normalize=False,
            nonnegative=True,
        )[0]

    return _chunked_coefficient_distance(
        first,
        second,
        histogram_converter,
        metric,
        ground_cost,
        chunk_size,
    )


def spherical_odf_distance(
    first,
    second,
    metric='total_variation',
    sphere=None,
    sh_order_max=8,
    chunk_size=1024,
):
    """Compare spherical ODF images using bounded histogram chunks."""
    if sphere is None:
        from dipy.data import get_sphere

        sphere = get_sphere(name='repulsion724')
    ground_cost = None
    if metric == 'wasserstein':
        vertices = np.asarray(sphere.vertices)
        ground_cost = np.arccos(np.clip(np.abs(vertices @ vertices.T), 0, 1))
    basis = _spherical_odf_basis(sphere, sh_order_max)

    def histogram_converter(coefficients):
        return _spherical_coefficients_to_histogram(
            coefficients,
            basis,
            normalize=False,
            nonnegative=True,
        )

    return _chunked_coefficient_distance(
        first,
        second,
        histogram_converter,
        metric,
        ground_cost,
        chunk_size,
    )