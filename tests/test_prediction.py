"""Tests for array-based prediction datasets."""

import numpy as np

from fibermetric.prediction import DiffusionTensorDataset


def test_diffusion_tensor_dataset_accepts_arrays():
    tensors = np.broadcast_to(np.eye(3), (2, 4, 5, 3, 3)).copy()
    mask = np.ones((2, 4, 5))
    mask[0] = 0

    dataset = DiffusionTensorDataset(tensors, mask)
    inputs, targets = dataset[1]

    assert len(dataset) == 2
    assert tuple(inputs.shape) == (3, 4, 5)
    assert tuple(targets.shape) == (4, 5)