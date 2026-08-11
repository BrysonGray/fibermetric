"""Tests for orientation direction representations."""

import numpy as np
import pytest

pytest.importorskip('dipy')

from fibermetric.orientation_encoding import spherical_odf


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


def test_spherical_odf_is_antipodally_symmetric_and_batched():
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    coefficients = spherical_odf(directions)
    antipodal_coefficients = spherical_odf(-directions)
    batched = spherical_odf(np.stack((directions, -directions)))

    assert batched.shape == (2, 45)
    assert np.allclose(coefficients, antipodal_coefficients)
    assert np.allclose(batched[0], batched[1])


def test_spherical_odf_rejects_invalid_directions():
    with pytest.raises(ValueError, match='nonzero'):
        spherical_odf(np.zeros((1, 3)))
    with pytest.raises(ValueError, match='nonnegative even'):
        spherical_odf(np.ones((1, 3)), sh_order_max=3)