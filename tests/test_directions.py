"""Tests for orientation direction representations."""

import importlib.util

import numpy as np
import pytest

from fibermetric.orientation_encoding.directions import circular_odf
from fibermetric.orientation_encoding.directions import circular_odf_directions
from fibermetric.orientation_encoding.directions import angle_to_rgb
from fibermetric.orientation_encoding.directions import project_to_plane
from fibermetric.orientation_encoding.directions import vec_to_theta
from fibermetric.orientation_encoding import spherical_odf
from fibermetric.orientation_encoding import spherical_odf_directions

DIPY_REQUIRED = pytest.mark.skipif(
    importlib.util.find_spec('dipy') is None,
    reason='DIPY is required for spherical ODF tests.',
)


def test_direction_helpers_preserve_broadcast_spatial_shape():
    vectors = np.ones((2, 3, 3))
    normals = np.zeros((2, 1, 3))
    normals[..., 2] = 1

    projected = project_to_plane(vectors, normals)
    rgb = angle_to_rgb(np.zeros((2, 3)), brightness=np.ones((2, 1)))
    theta = vec_to_theta(projected[..., :2])

    assert projected.shape == (2, 3, 3)
    assert np.allclose(projected[..., 2], 0)
    assert rgb.shape == (2, 3, 3)
    assert theta.shape == (2, 3)


def test_circular_odf_directions_finds_antipodal_axes_in_batches():
    expected = np.array([[0.2, -0.7], [0.4, -1.0]])
    coefficients = np.stack([
        circular_odf(angles, ntheta=720, decay=None)
        for angles in expected
    ])

    directions = circular_odf_directions(
        coefficients,
        max_directions=3,
        relative_threshold=0.5,
        ntheta=720,
        chunk_size=1,
    )

    assert directions.shape == (2, 3)
    for actual, wanted in zip(directions[:, :2], expected):
        differences = np.abs(np.angle(np.exp(2j * (actual[:, None] - wanted))))
        assert np.all(np.min(differences, axis=0) < 4 * np.pi / 720)
    assert np.isnan(directions[:, 2]).all()


def test_circular_odf_gathers_spatial_angles_to_output_shape():
    angles = np.zeros((4, 6))
    angles[:, 3:] = np.pi / 2

    coefficients = circular_odf(
        angles,
        shape_out=(2, 3),
        ntheta=72,
        n_coeffs=10,
        decay=None,
    )

    assert coefficients.shape == (2, 3, 10)
    assert not np.allclose(coefficients[:, 0], coefficients[:, 2])


@DIPY_REQUIRED
def test_spherical_odf_accepts_cartesian_and_spherical_directions():
    cartesian = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    spherical = np.array([[np.pi / 2, 0.0], [0.0, 0.0]])

    cartesian_coefficients = spherical_odf(cartesian, sh_order_max=8)
    spherical_coefficients = spherical_odf(
        spherical,
        coordinates='spherical',
        sh_order_max=8,
    )

    assert cartesian_coefficients.shape == (45,)
    assert np.allclose(cartesian_coefficients, spherical_coefficients)


@DIPY_REQUIRED
def test_spherical_odf_is_antipodally_symmetric_and_batched():
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    coefficients = spherical_odf(directions)
    antipodal_coefficients = spherical_odf(-directions)
    batched = np.stack((spherical_odf(directions), spherical_odf(-directions)))

    assert batched.shape == (2, 45)
    assert np.allclose(coefficients, antipodal_coefficients)
    assert np.allclose(batched[0], batched[1])


@DIPY_REQUIRED
def test_spherical_odf_gathers_spatial_vectors_to_output_shape():
    directions = np.zeros((4, 6, 3))
    directions[:, :3, 0] = 1
    directions[:, 3:, 2] = 1

    coefficients = spherical_odf(directions, shape_out=(2, 3))

    assert coefficients.shape == (2, 3, 45)
    assert not np.allclose(coefficients[:, 0], coefficients[:, 2])


@DIPY_REQUIRED
def test_spherical_odf_directions_finds_antipodal_axes_in_batches():
    expected = np.array([
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ])
    coefficients = spherical_odf(expected, shape_out=(2, 1))

    directions = spherical_odf_directions(
        coefficients,
        max_directions=3,
        relative_threshold=0.5,
        chunk_size=1,
    )

    assert directions.shape == (2, 1, 3, 3)
    for actual, wanted in zip(directions[:, 0, :2], expected):
        similarities = np.abs(actual @ wanted.T)
        assert np.all(np.max(similarities, axis=0) > 0.98)
    assert np.isnan(directions[:, :, 2]).all()


@DIPY_REQUIRED
def test_spherical_odf_rejects_invalid_directions():
    with pytest.raises(ValueError, match='nonzero'):
        spherical_odf(np.zeros((1, 3)))
    with pytest.raises(ValueError, match='nonnegative even'):
        spherical_odf(np.ones((1, 3)), sh_order_max=3)