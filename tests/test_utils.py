"""Tests for array utility functions."""

import numpy as np

from fibermetric.auxiliary import gather


def test_gather_crops_center_and_preserves_bin_order():
    values = np.arange(35).reshape(5, 7)

    gathered = gather(values, out_shape=(2, 3))

    assert gathered.shape == (2, 3, 4)
    assert np.array_equal(gathered[0, 0], values[0:2, 0:2].reshape(-1))
    assert np.array_equal(gathered[1, 2], values[2:4, 4:6].reshape(-1))


def test_gather_preserves_final_feature_axis():
    vectors = np.arange(4 * 6 * 3).reshape(4, 6, 3)

    gathered = gather(vectors, out_shape=(2, 3), feature_axis=-1)

    assert gathered.shape == (2, 3, 4, 3)
    assert np.array_equal(gathered[1, 2], vectors[2:4, 4:6].reshape(4, 3))