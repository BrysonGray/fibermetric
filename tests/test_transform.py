"""Tests for displacement-based image transforms."""

import numpy as np

from fibermetric.transform import sh_to_cf
from fibermetric.transform import ppd
from fibermetric.transform import transform_odf
from fibermetric.transform import transform_sh_img
from fibermetric.transform import transform_tensors_with_displacement


def test_transform_tensors_resamples_before_ppd():
    tensors = np.zeros((3, 2, 2, 3, 3))
    tensors[..., :, :] = np.diag([1.0, 2.0, 3.0])
    tensors *= np.arange(1, 4)[:, None, None, None, None]
    displacement = np.zeros((3, 3, 2, 2))
    displacement[0] = 1

    transformed = transform_tensors_with_displacement(tensors, displacement)

    assert np.allclose(transformed[:-1], tensors[1:], atol=1e-10)
    assert np.allclose(transformed[-1], 0, atol=1e-10)


def test_transform_sh_img_resamples_before_deformation():
    sh_image = np.zeros((3, 2, 2, 1))
    sh_image[..., 0] = np.arange(3)[:, None, None]
    displacement = np.zeros((3, 3, 2, 2))
    displacement[0] = 1

    transformed = transform_sh_img(sh_image, displacement)

    assert np.allclose(transformed[:-1], sh_image[1:], atol=1e-10)
    assert np.allclose(transformed[-1], 0, atol=1e-10)


def test_sh_to_cf_accepts_coefficient_last_images():
    sh_image = np.ones((2, 1))

    circular, azimuth = sh_to_cf(
        sh_image,
        ndir=20,
        nbins=12,
        sh_order_max=0,
    )

    assert circular.shape == (2, 12)
    assert azimuth.shape == (12,)
    assert np.allclose(circular.sum(axis=-1), 1)
    assert np.allclose(circular, circular[..., :1])


def test_ppd_broadcasts_leading_dimensions():
    tensors = np.broadcast_to(np.diag([1.0, 2.0, 3.0]), (2, 1, 3, 3))
    jacobian = np.broadcast_to(np.eye(3), (1, 4, 3, 3))

    rotations = ppd(tensors, jacobian)

    assert rotations.shape == (2, 4, 3, 3)
    assert np.allclose(rotations, np.eye(3))


def test_transform_odf_broadcasts_leading_dimensions():
    from dipy.data import get_sphere

    sphere = get_sphere(name='symmetric362')
    jacobian = np.broadcast_to(np.eye(3), (2, 1, 3, 3))
    odf = np.ones((1, 4, len(sphere.vertices)))

    transformed, vertices = transform_odf(jacobian, odf, sphere)

    assert transformed.shape == (2, 4, len(sphere.vertices))
    assert vertices.shape == (2, 4, len(sphere.vertices), 3)
    assert np.allclose(transformed, odf)
    assert np.allclose(vertices, sphere.vertices)