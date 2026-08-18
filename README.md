# fibermetric

A toolbox for extracting and comparing axon fiber orientations derived from
diffusion MRI and myelin histology.

## Command line interface

Run the package CLI with:

```bash
PYTHONPATH=src python -m fibermetric --help
```

Each implemented operation has a subcommand. Array-based commands use
NumPy `.npy` or `.npz` inputs and `.npy` outputs; commands returning multiple
arrays write `.npz` archives. When an input is an `.npz` archive, select the
array with `--key` (or `--first-key`, `--second-key`, `--tensors-key`,
`--displacement-key`).

### Orientation encoding

```bash
conda activate fibermetric
export PYTHONPATH=src

python -m fibermetric structure-tensor image.npy tensors.npy
python -m fibermetric principal-directions tensors.npy angles.npy
python -m fibermetric anisotropy tensors.npy anisotropy.npy --tensors

python -m fibermetric 2d-directions-to-circular-odf angles.npy circular_odf.npy --n-coeffs 16
python -m fibermetric 3d-directions-to-spherical-odf vectors.npy spherical_odf.npy --sh-order-max 8

python -m fibermetric circular-odf-to-histogram circular_odf.npy histogram.npz
python -m fibermetric spherical-odf-to-histogram spherical_odf.npy histogram_3d.npz

python -m fibermetric circular-odf-directions circular_odf.npy directions.npy
python -m fibermetric spherical-odf-directions spherical_odf.npy directions_3d.npy

python -m fibermetric circular-kmeans angles.npy centers.npy -k 2
python -m fibermetric spherical-kmeans vectors.npy centers_3d.npy -k 2
```

### Transforms

```bash
python -m fibermetric transform-tensors tensors.npy displacement.npy warped_tensors.npy
python -m fibermetric transform-sh spherical_odf.npy displacement.npy warped_odf.npy
python -m fibermetric sh-to-cf spherical_odf.npy circular_function.npz --bins 64
```

### Distances

```bash
python -m fibermetric odf-distance first.npy second.npy distance.npy \
	--representation circular --metric wasserstein
python -m fibermetric tensor-distance first.npy second.npy distance.npy \
	--metric riemannian
```

`--representation` and `--metric` are required for `odf-distance`; `--metric` is
required for `tensor-distance`.

### Phantoms, validation, and training

```bash
python -m fibermetric make-phantom angles.npy phantom.npy --shape 64 64 --spacing 1 1
python -m fibermetric run-sta-tests phantoms/ results/
python -m fibermetric train-unet training_data/ outputs/ --name model --epochs 100
```

Use `python -m fibermetric <command> --help` for command-specific options.

Computational APIs accept NumPy arrays and preserve leading image dimensions.
File loading and saving are confined to the CLI and `fibermetric.auxiliary.io`.
Tensor and SH transforms take 3D displacement arrays shaped `(3, X, Y, Z)`, with
the coordinate components on the first axis and values measured in voxel units.
Samples outside the source image are filled with zero.

ODF distances support `total_variation` and `wasserstein` metrics for circular
Fourier or spherical harmonic coefficients. Tensor distances support the
affine-invariant `riemannian` metric and averaged `symmetric_kl` divergence for
2D or 3D symmetric positive-definite tensors.
