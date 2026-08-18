"""Centralized input/output helpers."""

import os
from pathlib import Path

import numpy as np

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()


def load_array(path, key=None):
    """Load an array from a NumPy .npy or .npz file."""
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if key is None:
            if len(loaded.files) != 1:
                raise ValueError(f'{path} contains multiple arrays; specify a key.')
            key = loaded.files[0]
        return loaded[key]
    return loaded


def save_array(path, array):
    """Save an array to a NumPy .npy file, creating parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, array)


def save_arrays(path, **arrays):
    """Save named arrays to a NumPy .npz archive."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)


def load_image(path, key=None):
    """Load a NumPy array or a grayscale image based on file extension."""
    if Path(path).suffix.lower() in ('.npy', '.npz'):
        return load_array(path, key)
    return load_img(path)


def load_img(impath, img_down=0, reverse_intensity=False):
    """Load a grayscale histology image."""
    import cv2
    from skimage.transform import resize

    imname = os.path.split(impath)[1]
    print(f'loading image {imname}...')
    image = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
    if img_down:
        print('downsampling image...')
        image = resize(image, (image.shape[0] // img_down, image.shape[1] // img_down), anti_aliasing=True)
    if np.max(image[0]) > 1:
        image = image * 1 / 255
    if reverse_intensity is True:
        image = 1 - image
    return image


def read_data(path):
    """Read a NIfTI image and return coordinates, data, affine, and header."""
    import nibabel as nib

    image = nib.load(path)
    data = image.get_fdata()
    header = image.header
    affine = image.affine
    dim = header['dim'][1:data.ndim + 1]
    pixdim = header['pixdim'][1:data.ndim + 1]
    coords = []
    for axis, size in enumerate(dim):
        step = float(pixdim[axis])
        coords.append((np.arange(size) - (size - 1) / 2.0) * step)
    return coords, data, affine, header


def read_dti(dti_path):
    """Read a diffusion tensor volume from NIfTI."""
    coords, tensor_data, _, _ = read_data(dti_path)
    tensor_data = np.stack((
        tensor_data[..., 0], tensor_data[..., 3], tensor_data[..., 4],
        tensor_data[..., 3], tensor_data[..., 1], tensor_data[..., 5],
        tensor_data[..., 4], tensor_data[..., 5], tensor_data[..., 2],
    ), axis=-1)
    tensor_data = tensor_data.reshape(tensor_data.shape[:-1] + (3, 3))
    return coords, tensor_data


def load_odf(path):
    """Load ODF coefficient NIfTI volumes from a directory."""
    import nibabel as nib

    files = os.listdir(path)
    files.sort()
    signals = []
    for file_name in files:
        extension = os.path.splitext(file_name)[1]
        if extension in ('.nii', '.gz'):
            image = nib.load(os.path.join(path, file_name))
            signals.append(image.get_fdata())
    return np.array(signals)


def load_raw_dti(path, shape=(181, 6, 217, 181)):
    """Load a component-first raw diffusion tensor image."""
    components = np.fromfile(path, dtype=np.float32).reshape(shape)
    tensors = np.zeros((shape[0], shape[2], shape[3], 3, 3), dtype=np.float32)
    tensors[..., 0, 0] = components[:, 2]
    tensors[..., 1, 1] = components[:, 1]
    tensors[..., 2, 2] = components[:, 0]
    tensors[..., 0, 1] = tensors[..., 1, 0] = components[:, 5]
    tensors[..., 0, 2] = tensors[..., 2, 0] = components[:, 4]
    tensors[..., 1, 2] = tensors[..., 2, 1] = components[:, 3]
    return tensors


def load_raw_mask(path, shape=(181, 217, 181)):
    """Load a raw unsigned 16-bit image mask."""
    return np.fromfile(path, dtype=np.uint16).reshape(shape)