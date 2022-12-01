import numpy as np
import nibabel as nib
import argparse
from argparse import RawTextHelpFormatter
import os
from dipy.data import get_sphere
from dipy.core.sphere import Sphere
from dipy.reconst.shm import sh_to_sf
from tqdm import tqdm

def sh_to_cf(sh_signal, ndir, nbins, norm=True):
    
    """
    integration: add up the values of the function in each bin (holding phi constant),
    then multiply by the difference between boundaries (pi - 0 = pi)
    and divide by N-1 where N is the number of samples.

    Parameters
    ----------
    sh_signal : numpy array
        array of spherical harmonic coefficients.
        The first dimension must be coefficients, followed by spatial dimensions.
    ndir : int
        number of directions in theta from which to sample for integrating
    nbins : int
        number of bins for the function output
    scale : bool
        scales the circular functions to the interval [0,1].
    Returns
    -------
    cf : numpy array
        array of discrete functions on a circle with the first dimension being
        number of bins, followed by the shape of the input spatial dimensions.
    """
    
    step = 2*np.pi/nbins
    bins = np.arange(0,2*np.pi,step)
    cf = []
    # compute angles at which to sample
    phi = bins[None].T*np.ones((nbins, ndir))
    theta = np.arange(0, np.pi, np.pi/ndir) # uniformly spaced numbers from 0 to pi
    sh_flat = sh_signal.reshape((sh_signal.shape[0],-1)) # flatten the image (Ncoeffs, Nslices x Nrows x Ncols)
    dy = np.sin(theta) * np.pi/(ndir-1)
    cf = np.zeros((nbins,sh_flat.shape[1])) # (nbins, Nslices x Nrows x Ncols)

    for i in tqdm(range(nbins), desc=f'integrating...'):
        # sample from the spherical harmonic function
        x = Sphere(theta=theta, phi=phi[i]) # the input direction needs to be a dipy sphere object
        for j in range(sh_flat.shape[1]): # loop over every voxel
            y = sh_to_sf(sh_flat[:,j], x, sh_order=8) # sample from odf
            cf[i,j] = np.sum(y*dy) # integrate
    cf_shape = list(sh_signal.shape[1:])
    cf_shape.insert(0,nbins)
    cf = np.reshape(cf, tuple(cf_shape))
    if norm:
        cf = cf / np.sum(cf)

    return cf

def load_odf(path):
    files = os.listdir(path) 
    files.sort()
    sh_list = []
    for i in files:
        if os.path.splitext(i)[1] == '.gz':
            sh_img = nib.load(os.path.join(path, i))
            sh = sh_img.get_fdata()
            sh_list.append(sh)
    I = np.array(sh_list)

    return I

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
        description='Transforms an odf array described by spherical harmonics at each location into \n\
        an array of functions on a circle by integrating over the polar angle')
    parser.add_argument('-f', '--odf', help='Specify the folder containing spherical harmonic coefficients in nifti format', required=True)
    parser.add_argument('-d', '--density', help='Specify the sampling density for integrating over the polar angle as number of sampling directions.', default=100)
    parser.add_argument('-b', '--nbins', help='Specify the number of bins to discretize the circular function.', default=64)
    parser.add_argument('-o', '--output', help='Specify the output file name as the full path', required=True)
    parser.add_argument('-i', '--idx', help='Specify the first index in the flattened image array to process. (Optional)')
    parser.add_argument('-s', '--size', help='Specify the size of the batch of voxels to process')
    args = parser.parse_args()
    print(args)
    odf_path = args.odf
    ndirs = args.density
    nbins = args.nbins
    sh_signal = load_odf(odf_path)
    if args.idx:
        batch = [int(args.idx), int(args.idx)+int(args.size)]
        sh_signal = sh_signal.reshape((sh_signal.shape[0],-1))[:,batch[0]:batch[1]]
    cf = sh_to_cf(sh_signal, ndirs, nbins)
    cf_nii = nib.Nifti1Image(cf,affine=None)
    out = args.output
    if not os.path.exists(os.path.split(out)[0]):
        os.makedirs(os.path.split(out)[0])
    nib.save(cf_nii, args.output)
    
if __name__ == "__main__":
    main()