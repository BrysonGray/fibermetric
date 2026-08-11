"""Tests for the centralized command-line interface."""

import numpy as np

from fibermetric.cli import build_parser
from fibermetric.cli import main


def test_parser_exposes_checklist_commands():
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices
    expected = {
        'structure-tensor',
        'principal-directions',
        'anisotropy',
        'directions-to-odf',
        'directions-to-odf-3d',
        'odf-to-histogram',
        'spherical-kmeans',
        'circular-kmeans',
        'ppd',
        'transform-tensors',
        'sh-to-cf',
        'transform-sh',
        'make-phantom',
        'run-sta-tests',
        'train-unet',
        'tensor-distance',
        'odf-distance',
    }
    assert expected <= set(commands)


def test_orientation_pipeline(tmp_path):
    image_path = tmp_path / 'image.npy'
    tensors_path = tmp_path / 'tensors.npy'
    angles_path = tmp_path / 'angles.npy'
    coefficients_path = tmp_path / 'coefficients.npy'
    histogram_path = tmp_path / 'histogram.npz'
    np.save(image_path, np.eye(12))

    assert main(['structure-tensor', str(image_path), str(tensors_path)]) == 0
    assert main(['principal-directions', str(tensors_path), str(angles_path)]) == 0
    assert main([
        'directions-to-odf', str(angles_path), str(coefficients_path),
        '--n-coeffs', '10',
    ]) == 0
    assert main(['odf-to-histogram', str(coefficients_path), str(histogram_path)]) == 0

    assert np.load(tensors_path).shape == (12, 12, 2, 2)
    assert np.load(coefficients_path).shape == (10,)
    histogram = np.load(histogram_path)
    assert histogram['theta'].shape == histogram['histogram'].shape
    assert np.isclose(histogram['histogram'].sum(), 1.0)