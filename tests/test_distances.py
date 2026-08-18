"""Tests for ODF and tensor image distances."""

import numpy as np
import pytest

from fibermetric.difference_measures import circular_odf_distance
from fibermetric.difference_measures import apsym_vector_distance
from fibermetric.difference_measures import multiple_exclusive_distances
from fibermetric.difference_measures import periodic_distance_1d
from fibermetric.difference_measures import riemannian_tensor_distance
from fibermetric.difference_measures import spherical_odf_distance
from fibermetric.difference_measures import symmetric_kl_tensor_distance
from fibermetric.difference_measures import tensor_distance
from fibermetric.orientation_encoding import circular_odf
from fibermetric.orientation_encoding import spherical_odf


def test_periodic_distance_1d_wraps_across_multiple_periods():
    distances = periodic_distance_1d([0.0, 3.8], [0.2], period=1.0)

    assert distances.shape == (2,)
    assert np.allclose(distances, [0.2, 0.4])


def test_periodic_distance_1d_preserves_spatial_dimensions():
    first = np.zeros((3, 3))
    second = np.zeros((3, 3, 1))
    second[1, 2, 0] = np.pi / 2

    distances = periodic_distance_1d(first, second, period=np.pi)

    assert distances.shape == (3, 3)
    assert np.isclose(distances[1, 2], np.pi / 2)
    assert np.count_nonzero(distances) == 1


def test_periodic_distance_1d_broadcasts_leading_dimensions():
    first = np.array([[[0.0]], [[np.pi / 2]]])
    second = np.array([[[0.0], [np.pi / 2]]])

    distances = periodic_distance_1d(first, second, period=np.pi)

    assert distances.shape == (2, 2)
    assert np.allclose(distances, [[0.0, np.pi / 2], [np.pi / 2, 0.0]])


def test_apsym_vector_distance_is_elementwise_and_normalizes_vectors():
    distances = apsym_vector_distance(
        np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        np.array([[-3.0, 0.0, 0.0], [0.0, 0.0, 4.0]]),
    )

    assert np.allclose(distances, [0.0, 0.0])


def test_apsym_vector_distance_preserves_image_shape_for_2d_vectors():
    first = np.zeros((3, 3, 2))
    first[..., 0] = 1
    second = first.copy()
    second[1, 2] = [0, 1]

    distances = apsym_vector_distance(first, second)

    assert distances.shape == (3, 3)
    assert np.isclose(distances[1, 2], np.pi / 2)
    assert np.count_nonzero(distances) == 1


def test_apsym_vector_distance_broadcasts_leading_dimensions():
    first = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    second = np.array([[[1.0, 0.0], [0.0, 1.0]]])

    distances = apsym_vector_distance(first, second)

    assert distances.shape == (2, 2)
    assert np.allclose(distances, [[0.0, np.pi / 2], [np.pi / 2, 0.0]])


def test_apsym_vector_distance_accepts_single_vectors():
    distance = apsym_vector_distance([1.0, 0.0, 0.0], [-2.0, 0.0, 0.0])

    assert np.isclose(distance, 0.0)


def test_multiple_exclusive_distances_does_not_reuse_rows_or_columns():
    distances = np.array([[1.0, 4.0], [2.0, 3.0], [0.5, 5.0]])

    selected = multiple_exclusive_distances(distances)

    assert np.allclose(selected, [0.5, 3.0])


def test_circular_odf_distances_are_batched_and_antipodal():
    first = np.stack([
        circular_odf([0.0], ntheta=360, decay=None),
        circular_odf([0.0], ntheta=360, decay=None),
    ])
    second = np.stack([
        circular_odf([np.pi], ntheta=360, decay=None),
        circular_odf([np.pi / 2], ntheta=360, decay=None),
    ])

    total_variation = circular_odf_distance(
        first,
        second,
        metric='total_variation',
        ntheta=360,
        chunk_size=1,
    )
    wasserstein = circular_odf_distance(
        first,
        second,
        metric='wasserstein',
        ntheta=360,
        chunk_size=1,
    )

    assert np.allclose(total_variation, [0.0, 1.0])
    assert np.allclose(wasserstein, [0.0, np.pi / 2])


@pytest.mark.parametrize('metric', ['total_variation', 'wasserstein'])
def test_circular_odf_distances_broadcast_spatial_dimensions(metric):
    first = circular_odf([0.0], ntheta=72, decay=None)[None, None, :]
    second = np.stack([
        circular_odf([0.0], ntheta=72, decay=None),
        circular_odf([np.pi / 2], ntheta=72, decay=None),
    ])[None, :, :]

    distances = circular_odf_distance(
        first,
        second,
        metric=metric,
        ntheta=72,
    )

    assert distances.shape == (1, 2)
    assert np.isclose(distances[0, 0], 0)
    assert distances[0, 1] > 0


def test_spherical_odf_distances_are_batched_and_antipodal():
    first_directions = np.array([
        [[1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
    ])
    second_directions = np.array([
        [[-1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0]],
    ])
    first = spherical_odf(first_directions, shape_out=(2, 1))
    second = spherical_odf(second_directions, shape_out=(2, 1))

    total_variation = spherical_odf_distance(
        first,
        second,
        metric='total_variation',
        chunk_size=1,
    )
    wasserstein = spherical_odf_distance(
        first,
        second,
        metric='wasserstein',
        chunk_size=1,
    )

    assert total_variation.shape == (2, 1)
    assert np.isclose(total_variation[0, 0], 0)
    assert total_variation[1, 0] > 0
    assert np.isclose(wasserstein[0, 0], 0)
    assert wasserstein[1, 0] > 0


@pytest.mark.parametrize('metric', ['total_variation', 'wasserstein'])
def test_spherical_odf_distances_broadcast_spatial_dimensions(metric):
    first = spherical_odf(
        np.array([[1.0, 0.0, 0.0]]),
        shape_out=(1,),
    )[:, None, :]
    second = spherical_odf(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]), shape_out=(2,))[None, :, :]

    distances = spherical_odf_distance(first, second, metric=metric)

    assert distances.shape == (1, 2)
    assert np.isclose(distances[0, 0], 0)
    assert distances[0, 1] > 0


@pytest.mark.parametrize('dimensions', [2, 3])
def test_spd_tensor_distances_match_diagonal_analytic_values(dimensions):
    first = np.broadcast_to(np.eye(dimensions), (2, dimensions, dimensions)).copy()
    diagonal = np.ones(dimensions)
    diagonal[:2] = [np.e, np.exp(-1)]
    second = np.broadcast_to(np.diag(diagonal), first.shape).copy()
    second[0] = first[0]

    riemannian = riemannian_tensor_distance(first, second)
    symmetric_kl = symmetric_kl_tensor_distance(first, second)

    assert np.allclose(riemannian, [0, np.sqrt(2)])
    assert np.allclose(symmetric_kl, [0, 0.5 * (np.e + np.exp(-1) - 2)])


@pytest.mark.parametrize('metric', ['riemannian', 'symmetric_kl'])
def test_tensor_distances_broadcast_spatial_dimensions(metric):
    first = np.broadcast_to(np.eye(3), (2, 1, 3, 3))
    second = np.broadcast_to(np.eye(3), (1, 4, 3, 3))

    distances = tensor_distance(first, second, metric=metric)

    assert distances.shape == (2, 4)
    assert np.allclose(distances, 0)


def test_tensor_distances_reject_non_spd_inputs():
    with pytest.raises(ValueError, match='positive-definite'):
        riemannian_tensor_distance(np.eye(2), np.zeros((2, 2)))