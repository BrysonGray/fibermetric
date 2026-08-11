# fibermetric
## A toolbox for extracting and comparing axon fiber orientations derived from diffusion MRI and myelin histology.

## Command line interface

Run the package CLI with:

```bash
PYTHONPATH=src python -m fibermetric --help
```

Each implemented checklist operation has a subcommand. Array-based commands use
NumPy `.npy` inputs and outputs; commands returning multiple arrays write `.npz`
archives.

```bash
PYTHONPATH=src python -m fibermetric structure-tensor image.npy tensors.npy
PYTHONPATH=src python -m fibermetric principal-directions tensors.npy angles.npy
PYTHONPATH=src python -m fibermetric directions-to-odf angles.npy odf.npy --n-coeffs 16
PYTHONPATH=src python -m fibermetric odf-to-histogram odf.npy histogram.npz
```

Use `python -m fibermetric <command> --help` for command-specific options.
The `tensor-distance` and `odf-distance` commands are registered but report that
their underlying checklist functions are not implemented yet.
