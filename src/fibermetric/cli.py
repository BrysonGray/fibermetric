"""Command-line interface for fibermetric workflows."""

import argparse
from pathlib import Path

import numpy as np

from .auxiliary.io import load_array
from .auxiliary.io import load_image
from .auxiliary.io import save_array
from .auxiliary.io import save_arrays

# Orientation encoding commands
#------------------------------

def _run_structure_tensor(args):
    """Compute and save the structure tensors of a 2D or 3D image."""
    from .orientation_encoding.structure_tensor_analysis import structure_tensor

    result = structure_tensor(
        load_image(args.input, args.key),
        derivative_sigma=args.derivative_sigma,
        tensor_sigma=args.tensor_sigma,
        normalize=not args.no_normalize,
        masked=args.masked,
        id_minus_S=args.id_minus_s,
    )
    save_array(args.output, result)

def _run_principal_directions(args):
    """Compute and save the principal directions of a 2D or 3D structure tensor field."""
    from .orientation_encoding.structure_tensor_analysis import angles

    save_array(args.output, angles(load_array(args.input, args.key), cartesian=args.cartesian))


def _run_anisotropy(args):
    """Compute and save the anisotropy of a 2D or 3D eigenvalue or tensor field."""
    from .orientation_encoding.structure_tensor_analysis import anisotropy

    values = load_array(args.input, args.key)
    if args.tensors:
        values = np.linalg.eigvalsh(values)
    save_array(args.output, anisotropy(values))


def _run_2d_directions_to_circular_odf(args):
    """Compute and save the 2D ODF Fourier coefficients from principal directions."""
    from .orientation_encoding.directions import circular_odf

    coefficients = circular_odf(
        load_array(args.input, args.key),
        shape_out=args.shape_out,
        ntheta=args.ntheta,
        n_coeffs=args.n_coeffs,
        decay=args.decay,
        normalize=not args.no_normalize,
    )
    save_array(args.output, coefficients)


def _run_3d_directions_to_spherical_odf(args):
    """Compute and save 3D SH ODF coefficients from principal directions."""
    from .orientation_encoding.directions import spherical_odf

    coefficients = spherical_odf(
        load_array(args.input, args.key),
        coordinates=args.coordinates,
        shape_out=args.shape_out,
        sh_order_max=args.sh_order_max,
        normalize=not args.no_normalize,
    )
    save_array(args.output, coefficients)


def _run_circular_odf_to_histogram(args):
    """Compute and save the polar histogram from 2D ODF Fourier coefficients."""
    from .orientation_encoding.directions import circular_odf_to_histogram

    histogram, theta = circular_odf_to_histogram(
        load_array(args.input, args.key),
        ntheta=args.ntheta,
        normalize=not args.no_normalize,
        nonnegative=not args.allow_negative,
    )
    save_arrays(args.output, theta=theta, histogram=histogram)

def _run_spherical_odf_to_histogram(args):
    """Compute and save a sampled 3D histogram from SH ODF coefficients."""
    from .orientation_encoding.directions import spherical_odf_to_histogram

    histogram, sphere = spherical_odf_to_histogram(
        load_array(args.input, args.key),
        sh_order_max=args.sh_order_max,
        normalize=not args.no_normalize,
        nonnegative=not args.allow_negative,
    )
    save_arrays(args.output, histogram=histogram, vertices=np.asarray(sphere.vertices))


def _run_circular_odf_directions(args):
    """Compute and save principal axes from circular ODF coefficients."""
    from .orientation_encoding.directions import circular_odf_directions

    directions = circular_odf_directions(
        load_array(args.input, args.key),
        max_directions=args.max_directions,
        relative_threshold=args.relative_threshold,
        ntheta=args.ntheta,
        chunk_size=args.chunk_size,
    )
    save_array(args.output, directions)


def _run_spherical_odf_directions(args):
    """Compute and save principal axes from spherical ODF coefficients."""
    from .orientation_encoding.directions import spherical_odf_directions

    directions = spherical_odf_directions(
        load_array(args.input, args.key),
        max_directions=args.max_directions,
        relative_threshold=args.relative_threshold,
        sh_order_max=args.sh_order_max,
        chunk_size=args.chunk_size,
    )
    save_array(args.output, directions)

def _run_spherical_kmeans(args):
    from .orientation_encoding.periodic_kmeans import apsym_kmeans

    vectors = np.asarray(load_array(args.input, args.key), dtype=float)
    vectors = vectors.reshape((-1, vectors.shape[-1])).copy()
    if vectors.shape[-1] != 3:
        raise ValueError('spherical-kmeans input must contain 3D Cartesian vectors.')
    result = apsym_kmeans(vectors, k=args.clusters)
    if args.spherical:
        x_coord = result[..., 0]
        y_coord = result[..., 1]
        z_coord = result[..., 2]
        theta = np.arctan(np.sqrt(x_coord ** 2 + y_coord ** 2) / (z_coord + np.finfo(float).eps))
        theta = np.where(theta < 0, theta + np.pi, theta)
        phi = np.arctan(x_coord / (y_coord + np.finfo(float).eps))
        result = np.stack((theta, phi), axis=-1)
    save_array(args.output, result)


def _run_circular_kmeans(args):
    from .orientation_encoding.periodic_kmeans import periodic_kmeans

    result = periodic_kmeans(
        load_array(args.input, args.key),
        k=args.clusters,
        period=args.period,
        nstarts=args.starts,
    )
    save_array(args.output, result)


def _run_transform_tensors(args):
    from .transform import transform_tensors_with_displacement

    result = transform_tensors_with_displacement(
        load_array(args.tensors, args.tensors_key),
        load_array(args.displacement, args.displacement_key),
    )
    save_array(args.output, result)


def _run_sh_to_cf(args):
    from .transform import sh_to_cf

    signal = load_array(args.input, args.key)
    circular, azimuth = sh_to_cf(
        signal,
        ndir=args.directions,
        nbins=args.bins,
        normalize=not args.no_normalize,
        sh_order_max=args.sh_order_max,
    )
    save_arrays(args.output, circular_function=circular, azimuth=azimuth)


def _run_transform_sh(args):
    from .transform import transform_sh_img

    result = transform_sh_img(
        load_array(args.input, args.key),
        load_array(args.displacement, args.displacement_key),
    )
    save_array(args.output, result)


def _run_odf_distance(args):
    if args.representation == 'circular':
        from .difference_measures import circular_odf_distance

        result = circular_odf_distance(
            load_array(args.first, args.first_key),
            load_array(args.second, args.second_key),
            metric=args.metric,
            ntheta=args.ntheta,
            chunk_size=args.chunk_size,
        )
    else:
        from .difference_measures import spherical_odf_distance

        result = spherical_odf_distance(
            load_array(args.first, args.first_key),
            load_array(args.second, args.second_key),
            metric=args.metric,
            sh_order_max=args.sh_order_max,
            chunk_size=args.chunk_size,
        )
    save_array(args.output, result)


def _run_tensor_distance(args):
    from .difference_measures import tensor_distance

    result = tensor_distance(
        load_array(args.first, args.first_key),
        load_array(args.second, args.second_key),
        metric=args.metric,
    )
    save_array(args.output, result)


def _run_make_phantom(args):
    from .auxiliary.phantoms import make_phantom

    if len(args.shape) != len(args.spacing):
        raise ValueError('--shape and --spacing must have the same number of values.')
    coordinates = [np.arange(size) * spacing for size, spacing in zip(args.shape, args.spacing)]
    angles = load_array(args.angles)
    result = make_phantom(
        coordinates,
        angles,
        period=args.period,
        width=args.width,
        noise=args.noise,
        crop=args.crop,
        blur_correction=args.blur_correction,
        interp=not args.no_interp,
        inverse=args.inverse,
    )
    save_array(args.output, result)


def _run_sta_tests(args):
    from .auxiliary.sta_tests import run_from_files

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run_from_files(args.input, str(output))


def _run_train_unet(args):
    import torch

    from .auxiliary.io import load_raw_dti
    from .auxiliary.io import load_raw_mask
    from .prediction.unet import train_unet

    input_files = sorted(path for path in Path(args.input).iterdir() if path.is_file())
    tensor_images = [load_raw_dti(path) for path in input_files]
    masks = None
    if args.masks is not None:
        mask_files = sorted(path for path in Path(args.masks).iterdir() if path.suffix == '.img')
        masks = [load_raw_mask(path) for path in mask_files]
    model, losses, accuracies = train_unet(
        tensor_images,
        masks=masks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / f'{args.name}.pth')
    save_array(output / 'losses.npy', losses)
    save_array(output / 'accuracies.npy', accuracies)


def _add_array_io(parser, input_help='Input .npy or .npz file.'):
    parser.add_argument('input', help=input_help)
    parser.add_argument('output', help='Output .npy file.')
    parser.add_argument('--key', help='Array key when input is an .npz archive.')


def build_parser():
    """Build the fibermetric command-line parser."""
    parser = argparse.ArgumentParser(
        prog='fibermetric',
        description='Extract, transform, and compare fiber orientation representations.',
    )
    commands = parser.add_subparsers(dest='command', required=True)

    command = commands.add_parser('structure-tensor', help='Image -> structure tensor field.')
    _add_array_io(command, 'Input image, .npy, or .npz file.')
    command.add_argument('--derivative-sigma', type=float, default=1.0)
    command.add_argument('--tensor-sigma', type=float, default=1.0)
    command.add_argument('--no-normalize', action='store_true')
    command.add_argument('--masked', action='store_true')
    command.add_argument('--id-minus-s', action='store_true')
    command.set_defaults(handler=_run_structure_tensor)

    command = commands.add_parser('principal-directions', help='Tensor field -> principal directions.')
    _add_array_io(command)
    command.add_argument('--cartesian', action='store_true')
    command.set_defaults(handler=_run_principal_directions)

    command = commands.add_parser('anisotropy', help='Eigenvalues or tensors -> anisotropy.')
    _add_array_io(command)
    command.add_argument('--tensors', action='store_true', help='Treat input as a tensor field.')
    command.set_defaults(handler=_run_anisotropy)

    command = commands.add_parser('2d-directions-to-circular-odf', help='2D directions -> Fourier coefficients.')
    _add_array_io(command)
    command.add_argument('--shape-out', nargs='+', type=int, help='Spatial output shape; omit to pool all directions.')
    command.add_argument('--ntheta', type=int, default=500)
    command.add_argument('--n-coeffs', type=int)
    command.add_argument('--decay', type=float, default=0.1)
    command.add_argument('--no-normalize', action='store_true')
    command.set_defaults(handler=_run_2d_directions_to_circular_odf)

    command = commands.add_parser('3d-directions-to-spherical-odf', help='3D directions -> SH coefficients.')
    _add_array_io(command)
    command.add_argument('--shape-out', nargs='+', type=int, help='Spatial output shape; omit to pool all directions.')
    command.add_argument(
        '--coordinates',
        choices=('cartesian', 'spherical'),
        default='cartesian',
        help='Interpret input vectors as Cartesian xyz or spherical (polar, azimuth).',
    )
    command.add_argument('--sh-order-max', type=int, default=8)
    command.add_argument('--no-normalize', action='store_true')
    command.set_defaults(handler=_run_3d_directions_to_spherical_odf)

    command = commands.add_parser('circular-odf-to-histogram', help='Fourier coefficients -> polar histogram.')
    _add_array_io(command)
    command.add_argument('--ntheta', type=int, default=500)
    command.add_argument('--no-normalize', action='store_true')
    command.add_argument('--allow-negative', action='store_true')
    command.set_defaults(handler=_run_circular_odf_to_histogram)

    command = commands.add_parser('spherical-odf-to-histogram', help='SH coefficients -> sampled 3D histogram.')
    _add_array_io(command)
    command.add_argument('--sh-order-max', type=int, default=8)
    command.add_argument('--no-normalize', action='store_true')
    command.add_argument('--allow-negative', action='store_true')
    command.set_defaults(handler=_run_spherical_odf_to_histogram)

    command = commands.add_parser('circular-odf-directions', help='Circular ODF coefficients -> principal axes.')
    _add_array_io(command)
    command.add_argument('--max-directions', type=int, default=3)
    command.add_argument('--relative-threshold', type=float, default=0.1)
    command.add_argument('--ntheta', type=int, default=500)
    command.add_argument('--chunk-size', type=int, default=1024)
    command.set_defaults(handler=_run_circular_odf_directions)

    command = commands.add_parser('spherical-odf-directions', help='SH ODF coefficients -> principal axes.')
    _add_array_io(command)
    command.add_argument('--max-directions', type=int, default=3)
    command.add_argument('--relative-threshold', type=float, default=0.1)
    command.add_argument('--sh-order-max', type=int, default=8)
    command.add_argument('--chunk-size', type=int, default=1024)
    command.set_defaults(handler=_run_spherical_odf_directions)

    command = commands.add_parser('circular-kmeans', help='Cluster periodic 1D directions.')
    _add_array_io(command)
    command.add_argument('-k', '--clusters', type=int, required=True)
    command.add_argument('--period', type=float, default=np.pi)
    command.add_argument('--starts', type=int, default=1)
    command.set_defaults(handler=_run_circular_kmeans)

    command = commands.add_parser('spherical-kmeans', help='Cluster antipodally symmetric 3D directions.')
    _add_array_io(command)
    command.add_argument('-k', '--clusters', type=int, required=True)
    command.add_argument('--spherical', action='store_true', help='Return spherical coordinates.')
    command.set_defaults(handler=_run_spherical_kmeans)

    command = commands.add_parser('transform-tensors', help='Transform a DTI field with a displacement field.')
    command.add_argument('tensors', help='Input tensor .npy or .npz file.')
    command.add_argument('displacement', help='Input displacement .npy or .npz file.')
    command.add_argument('output', help='Output transformed tensor .npy file.')
    command.add_argument('--tensors-key', help='Tensor array key when input is an .npz archive.')
    command.add_argument('--displacement-key', help='Displacement array key when input is an .npz archive.')
    command.set_defaults(handler=_run_transform_tensors)

    command = commands.add_parser('transform-sh', help='Transform an SH image with a displacement field.')
    command.add_argument('input', help='Spherical harmonic image .npy or .npz file.')
    command.add_argument('displacement', help='Input displacement .npy or .npz file.')
    command.add_argument('output', help='Output transformed SH .npy file.')
    command.add_argument('--key', help='SH array key when input is an .npz archive.')
    command.add_argument('--displacement-key', help='Displacement array key when input is an .npz archive.')
    command.set_defaults(handler=_run_transform_sh)

    command = commands.add_parser('sh-to-cf', help='3D SH ODF -> 2D circular function.')
    _add_array_io(command)
    command.add_argument('--directions', type=int, default=100)
    command.add_argument('--bins', type=int, default=64)
    command.add_argument('--sh-order-max', type=int, default=8)
    command.add_argument('--no-normalize', action='store_true')
    command.set_defaults(handler=_run_sh_to_cf)

    command = commands.add_parser('make-phantom', help='Generate a structure-tensor validation phantom.')
    command.add_argument('angles', help='Angle array .npy file.')
    command.add_argument('output', help='Output phantom .npy file.')
    command.add_argument('--shape', nargs='+', type=int, required=True)
    command.add_argument('--spacing', nargs='+', type=float, required=True)
    command.add_argument('--period', type=float, default=10.0)
    command.add_argument('--width', type=float, default=1.0)
    command.add_argument('--noise', type=float, default=1e-6)
    command.add_argument('--crop', type=int)
    command.add_argument('--blur-correction', action='store_true')
    command.add_argument('--no-interp', action='store_true')
    command.add_argument('--inverse', action='store_true')
    command.set_defaults(handler=_run_make_phantom)

    command = commands.add_parser('run-sta-tests', help='Run structure-tensor validation tests.')
    command.add_argument('input', help='Input phantom .npz file or directory.')
    command.add_argument('output', help='Output directory.')
    command.set_defaults(handler=_run_sta_tests)

    command = commands.add_parser('train-unet', help='Train the out-of-plane orientation U-Net.')
    command.add_argument('input', help='Training data directory.')
    command.add_argument('output', help='Output directory.')
    command.add_argument('--name', required=True, help='Model output name.')
    command.add_argument('--masks', help='Optional mask directory.')
    command.add_argument('--epochs', type=int, default=100)
    command.add_argument('--batch-size', type=int, default=16)
    command.add_argument('--learning-rate', type=float, default=1e-4)
    command.add_argument('--random-seed', type=int, default=0)
    command.set_defaults(handler=_run_train_unet)

    command = commands.add_parser('odf-distance', help='Compare circular or spherical ODF images.')
    command.add_argument('first', help='First ODF .npy or .npz file.')
    command.add_argument('second', help='Second ODF .npy or .npz file.')
    command.add_argument('output', help='Output distance image .npy file.')
    command.add_argument('--first-key', help='First array key when input is an .npz archive.')
    command.add_argument('--second-key', help='Second array key when input is an .npz archive.')
    command.add_argument('--representation', choices=('circular', 'spherical'), required=True)
    command.add_argument('--metric', choices=('total_variation', 'wasserstein'), required=True)
    command.add_argument('--ntheta', type=int, default=500)
    command.add_argument('--sh-order-max', type=int, default=8)
    command.add_argument('--chunk-size', type=int, default=1024)
    command.set_defaults(handler=_run_odf_distance)

    command = commands.add_parser('tensor-distance', help='Compare SPD tensor images.')
    command.add_argument('first', help='First tensor .npy or .npz file.')
    command.add_argument('second', help='Second tensor .npy or .npz file.')
    command.add_argument('output', help='Output distance image .npy file.')
    command.add_argument('--first-key', help='First array key when input is an .npz archive.')
    command.add_argument('--second-key', help='Second array key when input is an .npz archive.')
    command.add_argument('--metric', choices=('riemannian', 'symmetric_kl'), required=True)
    command.set_defaults(handler=_run_tensor_distance)

    return parser


def main(argv=None):
    """Run the fibermetric command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ImportError, NotImplementedError, ValueError) as error:
        parser.error(str(error))
    return 0