"""Tensor and ODF image transformation routines."""

import numpy as np
from scipy.ndimage import map_coordinates

###############################
# Tensor-field transformations
###############################

def _validate_displacement(image, displacement):
    """Displacement must have one coordinate component per spatial dimension in its first axis, followed by those spatial dimensions; image must have channels in its first axis followed by the same spatial dimensions."""
    displacement = np.asarray(displacement, dtype=float)
    spatial_ndim = displacement.shape[0] if displacement.ndim else 0
    if spatial_ndim not in (2, 3):
        raise ValueError('displacement must have 2 or 3 components on its first axis.')
    if displacement.shape[1:] != image.shape[1:]:
        raise ValueError('displacement spatial dimensions must match the image.')
    return displacement, spatial_ndim


def _resample_with_displacement(image, displacement, pixelsize=1.0):
    """
    Arguments
    ---------
    image: array-like object of shape (c, h, w) or (c, h, w, d), where c is channels.
    displacement: array-like object of shape (2, h, w) or (3, h, w, d), where the first axis is the displacement vector components.
    pixelsize: float or sequence of floats, optional
        The spacing between pixels in each dimension. If a single float is provided, it is used for all dimensions. If a sequence is provided, it must have the same length as the number of spatial dimensions in the image. Default is 1.0.
    """
    image = np.asarray(image)
    displacement, spatial_ndim = _validate_displacement(image, displacement)
    pixelsize = np.array(pixelsize, dtype=float)
    if pixelsize.size == 1:
        pixelsize = np.full(spatial_ndim, pixelsize.item())
    elif pixelsize.size != spatial_ndim:
        raise ValueError('pixelsize must be a scalar or have one value per spatial dimension.')
    spatial_shape = displacement.shape[1:]
    coordinates = np.indices(spatial_shape, dtype=float) * pixelsize.reshape(
        (spatial_ndim,) + (1,) * spatial_ndim
    )
    coordinates += displacement
    n_channels = image.shape[0]
    resampled = np.empty((n_channels,) + spatial_shape, dtype=float)
    for channel in range(n_channels):
        resampled[channel] = map_coordinates(
            image[channel],
            coordinates,
            order=1,
            mode='constant',
            cval=0.0,
        )
    return resampled


def _displacement_jacobian(displacement):
    """displacement: array of shape (spatial_ndim, *spatial_shape) with coordinate components in the first axis."""
    displacement = np.asarray(displacement, dtype=float)
    spatial_ndim = displacement.shape[0]
    jacobian = np.zeros((spatial_ndim, spatial_ndim) + displacement.shape[1:])
    for component in range(spatial_ndim):
        derivatives = np.gradient(displacement[component])
        for axis, derivative in enumerate(derivatives):
            jacobian[component, axis] = derivative
    jacobian = np.moveaxis(jacobian, (0, 1), (-2, -1))
    jacobian += np.eye(spatial_ndim)
    return jacobian

def ppd(tensors, jacobian):
    """Preservation of principal directions transform."""
    tensors = np.asarray(tensors, dtype=float)
    jacobian = np.asarray(jacobian, dtype=float)
    if tensors.shape[-2:] != (3, 3) or jacobian.shape[-2:] != (3, 3):
        raise ValueError('tensors and jacobian must end with shape (3, 3).')
    try:
        spatial_shape = np.broadcast_shapes(tensors.shape[:-2], jacobian.shape[:-2])
    except ValueError as error:
        raise ValueError('tensors and jacobian must have broadcastable spatial dimensions.') from error
    tensors = np.broadcast_to(tensors, spatial_shape + (3, 3))
    jacobian = np.broadcast_to(jacobian, spatial_shape + (3, 3))

    _, eigenvectors = np.linalg.eigh(tensors)
    first = eigenvectors[..., -1]
    second = eigenvectors[..., -2]
    third = np.cross(first, second)
    source_frame = np.stack((first, second, third), axis=-1)

    target_first = np.einsum('...ij,...j->...i', jacobian, first)
    target_first /= np.linalg.norm(target_first, axis=-1, keepdims=True)
    target_second = np.einsum('...ij,...j->...i', jacobian, second)
    target_second -= np.sum(target_second * target_first, axis=-1, keepdims=True) * target_first
    target_second /= np.linalg.norm(target_second, axis=-1, keepdims=True)
    target_third = np.cross(target_first, target_second)
    target_frame = np.stack((target_first, target_second, target_third), axis=-1)

    return target_frame @ np.swapaxes(source_frame, -1, -2)


def transform_tensors_with_displacement(tensors, displacement, pixelsize=1.0):
    """Resample a 3D tensor image and reorient it using PPD."""
    tensors = np.asarray(tensors)
    if tensors.ndim < 2 or tensors.shape[-2:] != (3, 3):
        raise ValueError('tensors must have shape (X, Y, Z, 3, 3).')
    # move the (3, 3) tensor components to the first axis so tensors is channels-first.
    channels_first = np.moveaxis(tensors.reshape(tensors.shape[:-2] + (9,)), -1, 0)
    displacement, spatial_ndim = _validate_displacement(channels_first, displacement)
    if spatial_ndim != 3:
        raise ValueError('tensors must have shape (X, Y, Z, 3, 3).')
    resampled = _resample_with_displacement(channels_first, displacement, pixelsize)
    tensors = np.moveaxis(resampled, 0, -1).reshape(displacement.shape[1:] + (3, 3))
    jacobian = _displacement_jacobian(displacement)
    q_ppd = ppd(tensors, jacobian)
    return q_ppd @ tensors @ np.swapaxes(q_ppd, -1, -2)


#############################
# ODF transformations
#############################

# Spherical harmonics to circular function conversion
def sh_to_cf(sh_image, ndir=100, nbins=64, normalize=True, sh_order_max=8):
    """Integrate a coefficient-last SH ODF image over polar angle."""
    from dipy.core.sphere import Sphere
    from dipy.reconst.shm import sh_to_sf

    sh_image = np.asarray(sh_image)
    if sh_image.ndim < 1:
        raise ValueError('sh_image must be at least a 1D array.')
    if ndir < 2 or nbins < 1:
        raise ValueError('ndir must be at least 2 and nbins must be positive.')

    theta = np.linspace(0, np.pi, ndir)
    azimuth = np.arange(nbins) * (2 * np.pi / nbins)
    circular = np.empty(sh_image.shape[:-1] + (nbins,), dtype=float)
    for index, phi in enumerate(azimuth):
        sphere = Sphere(theta=theta, phi=np.full(ndir, phi))
        sampled = sh_to_sf(
            sh_image,
            sphere,
            sh_order_max=sh_order_max,
            basis_type='descoteaux07',
            legacy=False,
        )
        circular[..., index] = np.trapz(sampled * np.sin(theta), theta, axis=-1)

    if normalize:
        total = np.sum(circular, axis=-1, keepdims=True)
        circular = np.divide(
            circular,
            total,
            out=np.zeros_like(circular),
            where=np.abs(total) > np.finfo(float).eps,
        )
    return circular, azimuth


# ODF-field deformation
def transform_odf(jacobian, odf, sphere):
    """Apply Jacobian transforms to a discrete ODF image."""
    jacobian = np.asarray(jacobian, dtype=float)
    odf = np.asarray(odf, dtype=float)
    vertices = np.asarray(sphere.vertices, dtype=float)
    if jacobian.shape[-2:] != (3, 3):
        raise ValueError('jacobian must end with shape (3, 3).')
    if odf.ndim < 1 or odf.shape[-1] != len(vertices):
        raise ValueError('odf must end with one value per sphere vertex.')
    try:
        spatial_shape = np.broadcast_shapes(jacobian.shape[:-2], odf.shape[:-1])
    except ValueError as error:
        raise ValueError('jacobian and odf must have broadcastable spatial dimensions.') from error
    jacobian = np.broadcast_to(jacobian, spatial_shape + (3, 3))
    odf = np.broadcast_to(odf, spatial_shape + (len(vertices),))

    determinant = np.abs(np.linalg.det(jacobian))
    if np.any(determinant <= np.finfo(float).eps):
        raise ValueError('jacobian must be nonsingular.')

    transformed_vertices = np.einsum('...ij,vj->...vi', jacobian, vertices)
    radii = np.linalg.norm(transformed_vertices, axis=-1)
    transformed_vertices = transformed_vertices / radii[..., None]
    scale = radii ** 3 / determinant[..., None]
    return odf * scale, transformed_vertices


def transform_sh_img(sh, displacement, pixelsize=1.0):
    """Resample an SH image and deform its ODFs using a displacement field."""
    from dipy.core.geometry import cart2sphere
    from dipy.data import get_sphere
    from dipy.reconst.shm import real_sh_descoteaux, sh_to_sf_matrix

    sh = np.asarray(sh)
    order = int((np.sqrt(8 * sh.shape[-1] + 1) - 3) / 2)
    if (order + 1) * (order + 2) // 2 != sh.shape[-1] or order % 2:
        raise ValueError('The final image axis is not a symmetric SH coefficient count.')

    # move the SH coefficients to the first axis so sh is channels-first.
    channels_first = np.moveaxis(sh, -1, 0)
    displacement, spatial_ndim = _validate_displacement(channels_first, displacement)
    if spatial_ndim != 3:
        raise ValueError('spherical harmonic images require a 3D displacement field.')

    array_shape = displacement.shape[1:]
    resampled = np.moveaxis(
        _resample_with_displacement(channels_first, displacement, pixelsize), 0, -1
    ).reshape(-1, sh.shape[-1])
    jacobian = _displacement_jacobian(displacement).reshape(-1, 3, 3)
    determinant = np.abs(np.linalg.det(jacobian))
    if np.any(determinant <= np.finfo(float).eps):
        raise ValueError('the displacement jacobian must be nonsingular.')

    sphere = get_sphere(name='symmetric362')
    vertices = np.asarray(sphere.vertices, dtype=float)
    _, inverse_basis = sh_to_sf_matrix(
        sphere,
        sh_order_max=order,
        basis_type='descoteaux07',
        legacy=False,
    )
    inverse_jacobian = np.linalg.inv(jacobian)

    transformed = np.empty_like(resampled)
    for index in range(len(resampled)):
        # Pull the output directions back through the jacobian so the deformed ODF is
        # sampled on the fixed sphere; refitting on the deformed vertices is
        # ill-conditioned wherever the jacobian compresses them together.
        preimage = vertices @ inverse_jacobian[index].T
        radii = np.linalg.norm(preimage, axis=-1)
        preimage /= radii[:, None]
        _, theta, phi = cart2sphere(preimage[:, 0], preimage[:, 1], preimage[:, 2])
        basis, _, _ = real_sh_descoteaux(order, theta, phi, legacy=False)
        odf = (resampled[index] @ basis.T) / (determinant[index] * radii ** 3)
        transformed[index] = odf @ inverse_basis
    return transformed.reshape(array_shape + sh.shape[-1:])




