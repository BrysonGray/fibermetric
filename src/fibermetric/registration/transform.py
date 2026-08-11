"""Tensor-field registration routines."""

import numpy as np
import os
import pickle
import sympy
from sympy import Symbol
from sympy import Ynm
from sympy import integrate
from tqdm import tqdm

from ..auxiliary.utils import interp
from ..auxiliary.io import read_data
from ..auxiliary.io import read_dti


###############################
# Tensor-field transformations
###############################

def ppd(tensors, J):
    """Preservation of principal directions transform."""
    rot = lambda n, theta: np.array([
        [np.cos(theta) + n[..., 0, None] ** 2 * (1 - np.cos(theta)), n[..., 0, None] * n[..., 1, None] * (1 - np.cos(theta)) - n[..., 2, None] * np.sin(theta), n[..., 0, None] * n[..., 2, None] * (1 - np.cos(theta)) + n[..., 1, None] * np.sin(theta)],
        [n[..., 0, None] * n[..., 1, None] * (1 - np.cos(theta)) + n[..., 2, None] * np.sin(theta), np.cos(theta) + n[..., 1, None] ** 2 * (1 - np.cos(theta)), n[..., 1, None] * n[..., 2, None] * (1 - np.cos(theta)) - n[..., 0, None] * np.sin(theta)],
        [n[..., 0, None] * n[..., 2, None] * (1 - np.cos(theta)) - n[..., 1, None] * np.sin(theta), n[..., 1, None] * n[..., 2, None] * (1 - np.cos(theta)) + n[..., 0, None] * np.sin(theta), np.cos(theta) + n[..., 2, None] ** 2 * (1 - np.cos(theta))],
    ]).squeeze().transpose(2, 3, 4, 0, 1)

    _, e = np.linalg.eigh(tensors)
    e1 = e[..., -1]
    e2 = e[..., -2]
    je1 = np.squeeze(J @ e1[..., None])
    n1 = je1 / np.linalg.norm(je1, axis=-1)[..., None]
    je2 = np.squeeze(J @ e2[..., None])
    n2 = je2 / np.linalg.norm(je2, axis=-1)[..., None]
    theta = np.arccos(np.squeeze(e1[..., None, :] @ n1[..., None]))[..., None]
    r = np.cross(e1, n1) / np.sin(theta)
    theta2 = np.arccos(np.squeeze(e1[..., None, :] @ n2[..., None]))[..., None]
    r2 = np.cross(e1, n2) / np.sin(theta2)
    r[np.isnan(r)] = r2[np.isnan(r)]
    r1 = rot(r, theta)
    pn2 = n2 - (n2[..., None, :] @ n1[..., None])[..., 0] * n1
    pn2 = pn2 / np.linalg.norm(pn2, axis=-1)[..., None]
    r1e1 = np.squeeze(r1 @ e1[..., None])
    r1e2 = np.squeeze(r1 @ e2[..., None])
    phi = np.arccos(np.squeeze(r1e2[..., None, :] @ pn2[..., None]) / (np.linalg.norm(r1e2) * np.linalg.norm(pn2)))[..., None]
    r2m = rot(r1e1, phi)
    return r2m @ r1


def interp_dti(tensors, tensor_coords, points):
    """Interpolate a tensor field at target points."""
    if len(tensors.shape) != 5:
        raise Exception('T must contain 3x3 tensors or 6 diffusion components in last dimension')
    compact = np.stack((tensors[..., 0, 0], tensors[..., 1, 1], tensors[..., 2, 2], tensors[..., 0, 1], tensors[..., 0, 2], tensors[..., 1, 2]), -1)
    interpolated = []
    for index in range(6):
        interpolated.append(interp(tensor_coords, compact[..., index][None], points))
    interpolated = np.stack((interpolated[0], interpolated[3], interpolated[4], interpolated[3], interpolated[1], interpolated[5], interpolated[4], interpolated[5], interpolated[2]), axis=-1)
    return interpolated.reshape(interpolated.shape[:-1] + (3, 3)).squeeze()


def transform_tensors_with_displacement(tensor_path, displacement_path, original_path):
    """Apply a displacement field to a tensor volume using PPD."""
    _, displacement, _, _ = read_data(displacement_path)
    image_coords, _, _, _ = read_data(original_path)
    spacing = [(axis[1] - axis[0]) for axis in image_coords]
    mesh = np.stack(np.meshgrid(image_coords[0], image_coords[1], image_coords[2], indexing='ij'))
    transformed_points = displacement[0] + mesh
    tensor_coords, tensor_data = read_dti(tensor_path)
    tensors = interp_dti(tensor_data, tensor_coords, transformed_points)
    jacobian = np.stack(np.gradient(transformed_points[0], spacing[0], spacing[1], spacing[2], axis=(1, 2, 3))).transpose(2, 3, 4, 0, 1)
    q_ppd = ppd(tensors, jacobian)
    return q_ppd @ tensors @ q_ppd.transpose(0, 1, 2, 4, 3)


#############################
# ODF transformations
#############################

# Spherical harmonics to circular function conversion
def sh_to_cf_numeric(sh_signal, ndir, nbins, norm=True):
    """Integrate spherical harmonics numerically into a circular function."""
    from dipy.core.sphere import Sphere
    from dipy.reconst.shm import sh_to_sf

    step = 2 * np.pi / nbins
    bins = np.arange(0, 2 * np.pi, step)
    phi = bins[None].T * np.ones((nbins, ndir))
    theta = np.arange(0, np.pi, np.pi / ndir)
    sh_flat = sh_signal.reshape((sh_signal.shape[0], -1))
    dy = np.sin(theta) * np.pi / (ndir - 1)
    cf = np.zeros((nbins, sh_flat.shape[1]))
    for index in tqdm(range(nbins), desc='integrating...'):
        sphere = Sphere(theta=theta, phi=phi[index])
        for voxel in range(sh_flat.shape[1]):
            sampled = sh_to_sf(sh_flat[:, voxel], sphere, sh_order=8)
            cf[index, voxel] = np.sum(sampled * dy)
    shape = list(sh_signal.shape[1:])
    shape.insert(0, nbins)
    cf = np.reshape(cf, tuple(shape))
    if norm:
        cf = cf / np.sum(cf)
    return cf


def sh_to_cf(sh_signal, ndir, source):
    """Integrate ODF spherical harmonics over the polar angle."""
    theta = Symbol('theta')
    phi = Symbol('phi')
    degree = {1: 0, 2: 6, 15: 4, 28: 6, 45: 8, 66: 10, 91: 12, 120: 14}
    order = degree[sh_signal.shape[0]]
    try:
        path = os.path.join(source, 'Yphi.p')
        with open(path, 'rb') as handle:
            basis = pickle.load(handle)
            if len(basis) != sh_signal.shape[0]:
                raise Exception('basis does not match the spherical harmonic signal.')
    except Exception:
        basis = []
        for current in tqdm(np.arange(order + 1, step=2)):
            for degree_m in np.arange(-current, current + 1):
                if degree_m < 0:
                    basis.append(sympy.sqrt(2) * sympy.re(integrate(Ynm(current, degree_m, theta, phi).expand(func=True), (theta, 0, sympy.pi))))
                elif degree_m == 0:
                    basis.append(integrate(Ynm(current, degree_m, theta, phi).expand(func=True), (theta, 0, sympy.pi)))
                else:
                    basis.append(sympy.sqrt(2) * sympy.im(integrate(Ynm(current, degree_m, theta, phi).expand(func=True), (theta, 0, sympy.pi))))
        with open(os.path.join(source, 'Yphi.p'), 'wb') as handle:
            pickle.dump(basis, handle)
    matrix = np.zeros((ndir, len(basis)))
    sample_points = 0.0
    for direction in tqdm(range(ndir)):
        for index, value in enumerate(basis):
            sample_points = direction * 2 * np.pi / ndir
            matrix[direction, index] = float(value.evalf(subs={phi: sample_points}))
    cf = np.moveaxis(np.squeeze(matrix @ np.moveaxis(sh_signal, 0, -1)[..., None]), -1, 0)
    return cf, sample_points


# ODF-field deformation
def transform_odf(jacobian, odf, sphere):
    """Apply a Jacobian transform to a discrete ODF."""
    theta = sphere.theta
    phi = sphere.phi
    transformed_vertices = (sphere.vertices @ jacobian).T
    radius = np.linalg.norm(transformed_vertices)
    x_coord, y_coord, z_coord = transformed_vertices
    theta_transformed = np.arccos(z_coord / radius)
    j1 = np.stack((
        np.sin(theta) * np.cos(phi), np.cos(theta) * np.cos(phi), -np.sin(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi), np.cos(theta) * np.sin(phi), np.sin(theta) * np.cos(phi),
        np.cos(theta), -np.sin(theta), np.zeros(len(theta)),
    ), axis=-1).reshape((len(theta), 3, 3))
    j3 = np.stack((
        x_coord / radius, y_coord / radius, z_coord / radius,
        x_coord * z_coord / np.sqrt(x_coord ** 2 + y_coord ** 2 * radius ** 2), y_coord * z_coord / np.sqrt(x_coord ** 2 + y_coord ** 2 * radius ** 2), -np.sqrt(x_coord ** 2 + y_coord ** 2) / radius ** 2,
        -y_coord / (x_coord ** 2 + y_coord ** 2), x_coord / (x_coord ** 2 + y_coord ** 2), np.zeros(len(theta)),
    ), axis=-1).reshape((len(theta), 3, 3))
    polar = j3 @ jacobian @ j1
    scale = np.sin(theta) / np.sin(theta_transformed) * 1 / np.abs(np.linalg.det(polar[..., 1:, 1:]))
    return odf * scale, transformed_vertices


def transform_sh_img(sh, jacobian):
    """Transform a spherical harmonic image by a Jacobian field."""
    from dipy.data import get_sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    degree = {1: 0, 2: 6, 15: 4, 28: 6, 45: 8, 66: 10, 91: 12, 120: 14}
    order = degree[sh.shape[-1]]
    array_shape = sh.shape[:-1]
    sh = sh.reshape(-1, sh.shape[-1])
    jacobian = jacobian.reshape(-1, 3, 3)
    sphere = get_sphere('symmetric362')
    basis, inverse_basis = sh_to_sf_matrix(sphere, order)
    transformed = np.zeros_like(sh)
    for index in range(len(sh)):
        odf = np.dot(sh[index], basis)
        odf, vertices = transform_odf(jacobian[index], odf, sphere)
        basis, inverse_basis = sh_to_sf_matrix(vertices, order)
        transformed[index] = np.dot(inverse_basis.T, odf)
    return transformed.reshape(array_shape + sh.shape[-1:])




