#!/usr/bin/env python

import argparse
from argparse import RawTextHelpFormatter
import os
from diffusion import sh_to_cf
from utils import load_odf
import nibabel as nib

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