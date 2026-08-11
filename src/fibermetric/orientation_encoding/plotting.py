"""Orientation plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np

from .directions import angle_to_rgb
from .directions import circular_odf
from .directions import circular_odf_to_histogram
from .directions import vec_to_theta
from ..auxiliary.utils import draw


def plot_angles(image, angles=None, means=None, ntheta=500, axis=False, axes_coords=(0.1, 0.1, 0.8, 0.8), fig=None, show=True, title=None, xlabel=None, ylabel=None, colors=None):
    """Plot angles as a polar histogram over an image."""
    if fig is None:
        fig = plt.figure()

    ax_image = fig.add_axes(axes_coords)
    if len(image.shape) == 2:
        ax_image.imshow(image, cmap='gray', alpha=1)
    else:
        ax_image.imshow(image, alpha=1)
    plt.gca().set_yticklabels([])
    plt.gca().set_xticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    if title is not None:
        ax_image.set_title(title)
    if xlabel is not None:
        ax_image.set_xlabel(xlabel, fontsize=24)
    if ylabel is not None:
        ax_image.set_ylabel(ylabel, fontsize=24)

    if angles is not None:
        coefficients = circular_odf(angles, ntheta=ntheta, decay=0.1, normalize=False)
        theta, smoothed = circular_odf_to_histogram(coefficients, ntheta=ntheta, normalize=True, nonnegative=True)
        polar_coords = np.array(axes_coords) + np.array([0.05, 0.05, -0.1, -0.1])
        ax_polar = fig.add_axes(polar_coords, projection='polar')
        ax_polar.patch.set_alpha(0)
        ax_polar.plot(theta, smoothed, 'royalblue', linewidth=8)
        ax_polar.set_theta_offset(-np.pi / 2)
        ax_polar.set_yticklabels([])
        ax_polar.grid(False)
        if not axis:
            ax_polar.axis('off')

    if means is not None:
        if means.shape == ():
            means = [means]
        ax_means = fig.add_axes(axes_coords)
        for index, mean in enumerate(means):
            if colors is None:
                color = 'lime'
            elif isinstance(colors[index], (int, float)):
                color = angle_to_rgb(colors[index])
            else:
                color = np.abs(colors[index])
                if len(color) != 3:
                    raise ValueError('The colors must be either scalars or sequences of length 3.')
            if isinstance(mean, (int, float)):
                mean = [np.sin(mean), np.cos(mean)]
            ax_means.quiver(0, 0, mean[0], -mean[1], scale_units='width', scale=3, width=0.02, color=color)
            ax_means.quiver(0, 0, -mean[0], mean[1], scale_units='width', scale=3, width=0.02, color=color)
        ax_means.axis('off')

    if show:
        plt.show()


def plot_angles_3d(image, vectors=None, means=None, mip=False):
    """Plot 3D vector orientations in orthogonal image views."""
    if mip:
        orthogonal = [image.max(axis=0), image.max(axis=1), image.max(axis=2)]
    else:
        orthogonal = [image[image.shape[0] // 2], image[:, image.shape[1] // 2], image[:, :, image.shape[2] // 2]]

    if vectors is not None:
        vectors = vectors.reshape(-1, 3)
        x_coord = vectors[..., 0]
        y_coord = vectors[..., 1]
        z_coord = vectors[..., 2]
        vectors_2d = [
            np.stack((x_coord, y_coord), axis=-1),
            np.stack((x_coord, z_coord), axis=-1),
            np.stack((y_coord, z_coord), axis=-1),
        ]
        angles_2d = [vec_to_theta(vector) for vector in vectors_2d]
    else:
        angles_2d = [None, None, None]

    if means is not None:
        means_2d = []
        for mean in means:
            means_2d.append([[mean[0], mean[1]], [mean[0], mean[2]], [mean[1], mean[2]]])
        means_2d = np.transpose(np.array(means_2d), (1, 0, 2))
        colors = np.abs(means)
    else:
        means_2d = [None, None, None]
        colors = None

    fig = plt.figure()
    axes_coords_list = [[0.1, 0.1, 0.8, 0.8], [1.0, 0.1, 0.8, 0.8], [2.0, 0.1, 0.8, 0.8]]
    xlabels = ['X', 'X', 'Y']
    ylabels = ['Y', 'Z', 'Z']

    for index in range(3):
        plot_angles(
            image=orthogonal[index],
            angles=angles_2d[index],
            means=means_2d[index],
            axes_coords=axes_coords_list[index],
            fig=fig,
            show=False,
            title=None,
            xlabel=xlabels[index],
            ylabel=ylabels[index],
            colors=colors,
        )

    plt.show()


def visualize(tensors, tensor_coords, **kwargs):
    """Visualize a tensor volume using principal eigenvector RGB encoding."""
    eigenvalues, eigenvectors = np.linalg.eigh(tensors)
    principal = eigenvectors[..., -1]
    image = np.abs(principal).transpose(3, 0, 1, 2)
    trace = np.trace(tensors, axis1=-2, axis2=-1)
    eigenvalues = eigenvalues.transpose(3, 0, 1, 2)
    anisotropy = np.sqrt((3 / 2) * (np.sum((eigenvalues - (1 / 3) * trace) ** 2, axis=0) / np.sum(eigenvalues ** 2, axis=0)))
    anisotropy = np.nan_to_num(anisotropy)
    anisotropy = anisotropy / np.max(anisotropy)
    image = image * anisotropy
    draw(image, xJ=tensor_coords, **kwargs)
    return image