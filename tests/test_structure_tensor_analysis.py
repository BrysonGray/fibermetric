"""Tests for structure tensor analysis shape semantics."""

import numpy as np

from fibermetric.orientation_encoding.structure_tensor_analysis import anisotropy
from fibermetric.orientation_encoding.structure_tensor_analysis import hsv


def test_anisotropy_preserves_arbitrary_leading_dimensions():
    eigenvalues = np.ones((2, 3, 4, 3))
    eigenvalues[..., 2] = 2

    values = anisotropy(eigenvalues)

    assert values.shape == (2, 3, 4)
    assert np.isfinite(values).all()


def test_hsv_preserves_three_spatial_dimensions():
    tensors = np.broadcast_to(np.diag([1.0, 2.0]), (2, 3, 4, 2, 2))
    image = np.ones((4, 6, 8))

    theta, anisotropy_values, rgb = hsv(tensors, image)

    assert theta.shape == (2, 3, 4)
    assert anisotropy_values.shape == (2, 3, 4)
    assert rgb.shape == (2, 3, 4, 3)