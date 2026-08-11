#!/usr/bin/env python
# ruff: noqa: E501
# ruff: noqa: E741
"""Structure tensor validation routines."""

import numpy as np
import pandas as pd
from tqdm.contrib import itertools as tqdm_itertools

from . import periodic_kmeans
from . import utils
from .phantoms import make_phantom
from ..orientation_encoding import angles as compute_angles
from ..orientation_encoding import structure_tensor


def sta_test(I, derivative_sigma, tensor_sigma, true_thetas=None, crop=None, crop_end=None):
    """Test structure tensor analysis on a phantom image."""
    nI = I.shape
    dim = len(nI)
    if dim == 0 or dim > 3:
        raise TypeError(f"Input image should have two or three dimensions but got {dim}")
    if len(true_thetas) == 0 or len(true_thetas) > 3:
        raise Exception(f"Argument \"true_thetas\" must be have length 1 or 2 but got {len(true_thetas)}.")

    tensors = structure_tensor(I, derivative_sigma=derivative_sigma, tensor_sigma=tensor_sigma)
    if dim == 2:
        angle_values = compute_angles(tensors)
    else:
        angle_values = compute_angles(tensors, cartesian=True)

    if crop == 0.0:
        crop = None
    if crop_end == 0.0:
        crop_end = None
    if crop is not None and crop > 0:
        angle_values = angle_values[crop:-crop, crop:-crop]
    if crop_end is not None:
        angle_values = angle_values[:-crop_end]

    if dim == 2:
        angle_values = angle_values.flatten()
        angle_values = np.where(angle_values < 0, angle_values + np.pi, angle_values)
        if len(true_thetas) == 1:
            x = np.arange(180) * np.pi / 180
            mu = periodic_kmeans.periodic_mean(angle_values, x, period=np.pi)[None]
        else:
            mu = periodic_kmeans.periodic_kmeans(angle_values, period=np.pi, k=2)
        diff = periodic_kmeans.distance(mu, np.array(true_thetas), period=np.pi)
        diff = periodic_kmeans.multiple_exclusive_distances(diff)
        error = np.mean(diff)
    else:
        angle_values = angle_values.reshape(-1, dim)
        true_thetas = utils.sph_to_cart(true_thetas)
        if len(true_thetas) == 1:
            mu = periodic_kmeans.apsym_kmeans(angle_values, k=1)
            diff = np.arccos(np.abs(mu.dot(true_thetas.T)))
        else:
            mu = periodic_kmeans.apsym_kmeans(angle_values, k=2)
            diff = periodic_kmeans.distance_3d(mu, true_thetas)
            diff = periodic_kmeans.multiple_exclusive_distances(diff)
            diff = np.mean(diff)
        error = np.mean(diff)

    return error.astype(np.float64) * 180 / np.pi


def run_tests(derivative_sigmas, tensor_sigmas, nIs, angles, periods=[10], blur_correction=False):
    """Run a series of structure tensor validation tests."""
    error_df = pd.DataFrame({'derivative_sigma': [], 'tensor_sigma': [], 'anisotropy_ratio': [], 'period': [], 'angles': [], 'error': []})
    if not isinstance(derivative_sigmas, (list, tuple, np.ndarray)):
        derivative_sigmas = [derivative_sigmas]
    if not isinstance(tensor_sigmas, (list, tuple, np.ndarray)):
        tensor_sigmas = [tensor_sigmas]
    if not isinstance(nIs[0], (list, tuple, np.ndarray)):
        nIs = [nIs]
    if not isinstance(periods, (list, tuple, np.ndarray)):
        periods = [periods]
    if not isinstance(angles[0], (list, tuple, np.ndarray)):
        angles = [angles]

    for i1, i2, i3 in tqdm_itertools.product(range(len(nIs)), range(len(periods)), range(len(angles))):
        nI = nIs[i1]
        anisotropy_ratio = float(nI[1] / nI[0])
        if len(nI) == 2:
            dI = (anisotropy_ratio, 1.0)
        elif len(nI) == 3:
            dI = (anisotropy_ratio, 1.0, 1.0)
        else:
            raise Exception(f"nI must have length of either two or three but got {len(nI)}")
        x = [np.arange(ni) * di for ni, di in zip(nI, dI)]
        period = periods[i2]
        angle = angles[i3]
        image = make_phantom(x, angle, period, blur_correction=blur_correction)
        for derivative_sigma in derivative_sigmas:
            for tensor_sigma in tensor_sigmas:
                crop_all = round(max(derivative_sigma, tensor_sigma) * 8 / 3)
                crop_end = round(anisotropy_ratio) - 1
                error = sta_test(image, derivative_sigma, tensor_sigma, true_thetas=angle, crop=crop_all, crop_end=crop_end)
                new_row = {
                    'derivative_sigma': derivative_sigma,
                    'tensor_sigma': tensor_sigma,
                    'anisotropy_ratio': anisotropy_ratio,
                    'period': period,
                    'angles': [angle],
                    'error': error,
                }
                error_df = pd.concat((error_df, pd.DataFrame(new_row)), ignore_index=True)
    return error_df