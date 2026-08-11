#!/usr/bin/env python
# ruff: noqa: E501
"""Structure tensor phantom generation utilities."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from . import utils


def make_phantom(x, angles, period=10, width=1.0, noise=1e-6, crop=None, blur_correction=False, display=False, interp=True, inverse=False):
    """Generate a 2D or 3D line phantom for structure tensor validation."""
    d = np.array([xi[1] - xi[0] for xi in x])
    b = np.array([len(xi) // 2 for xi in x])
    X = np.stack(np.meshgrid(*x, indexing='ij'), axis=-1)
    blur_factor = np.sqrt(d[0] ** 2 - d[1] ** 2)

    image = np.random.randn(*X.shape[:-1]) * noise

    if len(x) == 3:
        sigma = (np.diag(d) * width) ** 2
        blur = (0.0, blur_factor, blur_factor)
        for angle in angles:
            direction = utils.sph_to_cart(angle, order='ij')
            if np.all(direction == [1.0, 0.0, 0.0]):
                sigma_ = sigma
                x_ = (X - b)[..., None]
            else:
                axis = np.cross(direction, np.array([1.0, 0.0, 0.0]))
                axis = axis / np.sum(axis ** 2) ** 0.5
                alpha = np.arccos(np.dot(direction, np.array([1.0, 0.0, 0.0])))
                K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
                rotation = expm(alpha * K)
                sigma_ = rotation @ sigma @ rotation.T
                x_ = (rotation @ (X - b)[..., None])
            sigma__ = sigma_[1:, 1:]
            norm = 1.0 / np.sqrt(2.0 * np.pi ** 2) / np.linalg.det(sigma__) ** 0.5
            x__ = x_[..., 1:, :]
            if period is not None:
                x__ = ((x__ + period / 2) % period) - period / 2
            tmp = np.linalg.inv(sigma__) @ x__
            tmp = x__.swapaxes(-1, -2) @ tmp
            image += norm * np.exp(-0.5 * tmp[..., 0, 0])
        if inverse:
            image = np.exp(-10 * image)
        if blur_correction:
            image = utils.anisotropy_correction(image, d, blur=blur)
        elif interp:
            image = utils.anisotropy_correction(image, d)
        if crop is not None and crop > 0:
            image[crop:-crop, crop:-crop, crop:-crop]
        if display:
            fig, ax = plt.subplots(3, figsize=(6, 4))
            ax[0].imshow(image[image.shape[0] // 2])
            ax[0].set_title('Image xy')
            ax[1].imshow(image[:, image.shape[1] // 2])
            ax[1].set_title('Image zx')
            ax[2].imshow(image[:, :, image.shape[2] // 2])
            ax[2].set_title('Image zy')
            plt.show()
        return image

    if len(x) == 2:
        blur = (0.0, blur_factor)
        for angle in angles:
            sigma = (np.sin(angle) * d[0] * width) ** 2 + (np.cos(angle) * d[1] * width) ** 2
            x__ = (X - b) @ np.array([-np.sin(angle), np.cos(angle)])
            if period is not None:
                x__ = ((x__ + period / 2) % period) - period / 2
            norm = 1.0 / (2.0 * np.pi * sigma)
            image += norm * np.exp(-0.5 * x__ ** 2 / sigma)
        if inverse:
            image = np.exp(-10 * image)
        if blur_correction:
            image = utils.anisotropy_correction(image, d, blur=True)
        elif interp:
            image = utils.anisotropy_correction(image, d)
        if crop is not None and crop > 0:
            image[crop:-crop, crop:-crop]
        if display:
            plt.imshow(image)
            plt.title('Image')
        return image

    return image