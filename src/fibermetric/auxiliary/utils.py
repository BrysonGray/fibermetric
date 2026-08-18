#!/usr/bin/env python

"""Utils.py : Helper functions."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn
from typing import Literal
import scipy

from .io import read_data


def sph_to_cart(x: np.ndarray, order: Literal['ij', 'xy'] = 'xy') -> np.ndarray:
    if x.ndim == 1:
        x = x[None]
    if order=='xy':
        xout = np.array([np.sin(x[:,0])*np.sin(x[:,1]),
                                np.sin(x[:,0])*np.cos(x[:,1]),
                                np.cos(x[:,0])
                                ]).T
    elif order=='ij':
        xout = np.array([np.cos(x[:,0]),
                        np.sin(x[:,0])*np.cos(x[:,1]),
                        np.sin(x[:,0])*np.sin(x[:,1])
                        ]).T
    
    return xout.squeeze()


def anisotropy_correction(image, dI, direction='up', blur=False):
    isotropic = np.all(np.array(dI) == dI[0])
    if not isotropic:
    # downsample all dimensions to largest dimension or upsample to the smallest dimension.
        x_in = [np.arange(n)*d for n,d in zip(image.shape, dI)]

        if direction == 'down':
            dx = np.max(dI)
        elif direction == 'up':
            dx = np.min(dI)

        x_out = [np.arange(0,n*d, step=dx) for n, d in zip(image.shape, dI)]
        Xout = np.stack(np.meshgrid(*x_out, indexing='ij'), axis=-1)
        image = scipy.interpolate.interpn(points=x_in, values=image, xi=Xout, method='linear', bounds_error=False, fill_value=None)
        
    if blur is not False:
        image = gaussian_filter(image, sigma=blur)
    
    return image


def interp(grid, values, points):
    """Interpolate values defined on a regular grid at target points."""
    moved = np.moveaxis(points, 0, -1)
    return interpn(points=grid, values=np.squeeze(values), xi=moved, method='linear', bounds_error=False, fill_value=None)


def draw(image, xJ=None, **kwargs):
    """Minimal visualization helper for tensor and orientation outputs."""
    if image.ndim == 4 and image.shape[0] in (1, 3, 4):
        display = np.moveaxis(image, 0, -1)
        plt.imshow(np.squeeze(display), **kwargs)
    elif image.ndim == 3:
        plt.imshow(np.squeeze(image), **kwargs)
    else:
        plt.imshow(image, **kwargs)
    if xJ is not None:
        plt.title('visualization')
    return plt.gca()


def gather(values, out_shape, feature_axis=None):
    """Gather a spatial array into equal non-overlapping bins.

    Spatial edges are cropped symmetrically when needed. Set ``feature_axis``
    to ``-1`` to preserve a final feature axis after the gathered sample axis.
    """
    values = np.asarray(values)
    out_shape = tuple(int(size) for size in out_shape)
    if feature_axis not in (None, -1):
        raise ValueError('feature_axis must be None or -1.')

    spatial_shape = values.shape if feature_axis is None else values.shape[:-1]
    if len(out_shape) != len(spatial_shape):
        raise ValueError('out_shape must have one value per spatial dimension.')
    if any(size < 1 for size in out_shape):
        raise ValueError('out_shape values must be positive.')

    bin_shape = tuple(size // output for size, output in zip(spatial_shape, out_shape))
    if any(size < 1 for size in bin_shape):
        raise ValueError('out_shape cannot exceed the input spatial shape.')
    cropped_shape = tuple(output * size for output, size in zip(out_shape, bin_shape))
    starts = tuple((size - cropped) // 2 for size, cropped in zip(spatial_shape, cropped_shape))
    slices = tuple(slice(start, start + cropped) for start, cropped in zip(starts, cropped_shape))
    if feature_axis == -1:
        slices += (slice(None),)
    cropped = np.ascontiguousarray(values[slices])

    interleaved_shape = tuple(
        dimension
        for output, size in zip(out_shape, bin_shape)
        for dimension in (output, size)
    )
    if feature_axis == -1:
        interleaved_shape += values.shape[-1:]
    gathered = cropped.reshape(interleaved_shape)

    spatial_axes = tuple(range(0, 2 * len(out_shape), 2))
    sample_axes = tuple(range(1, 2 * len(out_shape), 2))
    axes = spatial_axes + sample_axes
    if feature_axis == -1:
        axes += (2 * len(out_shape),)
    gathered = gathered.transpose(axes)
    sample_count = int(np.prod(bin_shape))
    if feature_axis == -1:
        return gathered.reshape(out_shape + (sample_count, values.shape[-1]))
    return gathered.reshape(out_shape + (sample_count,))