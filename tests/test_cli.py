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
        '2d-directions-to-circular-odf',
        '3d-directions-to-spherical-odf',
        'circular-odf-to-histogram',
        'spherical-odf-to-histogram',
        'circular-odf-directions',
        'spherical-odf-directions',
        'spherical-kmeans',
        'circular-kmeans',
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
        '2d-directions-to-circular-odf', str(angles_path), str(coefficients_path),
        '--n-coeffs', '10',
        '--shape-out', '3', '3',
    ]) == 0
    assert main(['circular-odf-to-histogram', str(coefficients_path), str(histogram_path)]) == 0

    assert np.load(tensors_path).shape == (12, 12, 2, 2)
    assert np.load(coefficients_path).shape == (3, 3, 10)
    histogram = np.load(histogram_path)
    assert histogram['histogram'].shape == (3, 3, histogram['theta'].size)
    assert np.allclose(histogram['histogram'].sum(axis=-1), 1.0)


def test_circular_odf_directions_command(tmp_path):
    coefficients_path = tmp_path / 'coefficients.npy'
    directions_path = tmp_path / 'directions.npy'
    coefficients = np.zeros(10, dtype=complex)
    coefficients[0] = 1
    np.save(coefficients_path, coefficients)

    assert main([
        'circular-odf-directions',
        str(coefficients_path),
        str(directions_path),
        '--max-directions',
        '2',
    ]) == 0

    assert np.load(directions_path).shape == (2,)


def test_odf_distance_command(tmp_path):
    first_path = tmp_path / 'first.npy'
    second_path = tmp_path / 'second.npy'
    output_path = tmp_path / 'distance.npy'
    coefficients = np.zeros((2, 10), dtype=complex)
    coefficients[:, 0] = 1
    np.save(first_path, coefficients)
    np.save(second_path, coefficients)

    assert main([
        'odf-distance',
        str(first_path),
        str(second_path),
        str(output_path),
        '--representation',
        'circular',
        '--metric',
        'total_variation',
    ]) == 0

    assert np.allclose(np.load(output_path), np.zeros(2))


def test_tensor_distance_command(tmp_path):
    first_path = tmp_path / 'first.npy'
    second_path = tmp_path / 'second.npy'
    output_path = tmp_path / 'distance.npy'
    tensors = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    np.save(first_path, tensors)
    np.save(second_path, tensors)

    assert main([
        'tensor-distance',
        str(first_path),
        str(second_path),
        str(output_path),
        '--metric',
        'riemannian',
    ]) == 0

    assert np.allclose(np.load(output_path), np.zeros(2))