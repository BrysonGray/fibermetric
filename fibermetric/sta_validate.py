#!/usr/bin/env python

'''
Structure tensor analysis validation functions.

Author: Bryson Gray
2023

'''

import sys
sys.path.insert(0, '/home/brysongray/fibermetric/')
from fibermetric import histology
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,sobel,correlate1d,gaussian_filter1d
import cv2


def radial_lines_2d(thetas: tuple, nI: tuple[int], dI: tuple[float], width: int= 2, noise: float=0.1, blur=1.0, mask_thresh: float=0.1):

    xI = [(np.arange(n) - (n-1)/2)*d for n,d in zip(nI,dI)]
    xmax = xI[1][-1]
    ymax = xI[0][-1]
    np.random.seed(1)
    I = np.random.randn(*nI)*noise
    mask = np.zeros_like(I)
    labels = np.zeros_like(I)
    worldtogrid = lambda p,dI,nI : tuple(np.round(x/d + (n-1)/2).astype(int) for x,d,n in zip(p,dI,nI))
    for i in range(len(thetas)):
        theta = thetas[i]
        m = np.tan(theta)
        y0 = m*xmax
        y1 = -y0
        x0 = ymax/(m+np.finfo(float).eps)
        x1 = -x0
        if np.abs(x0) > xmax:
            x0 = xmax
            x1 = -xmax
        elif np.abs(y0) > ymax:
            y0 = ymax
            y1 = -ymax
        j0,i0 = worldtogrid((y0,x0),dI,nI)
        j1,i1 = worldtogrid((y1,x1),dI,nI)
        start_point = (i0,j0)
        end_point = (i1,j1)
        I_ = cv2.line(np.zeros_like(I),start_point,end_point,thickness=width, color=(1))
        mask += I_
        labels += np.where(I_==1.0, theta*(180/np.pi), 0.0)
        I += I_
    mask = np.where(mask==1.0, mask, 0.0)
    labels = labels*mask
    I = np.where(I > 1.2, 1.0, I)
    blur = [blur/dI[0],blur/dI[1]]
    I = gaussian_filter(I,sigma=blur)
    extent = (xI[1][0]-dI[1]/2, xI[1][-1]+dI[1]/2, xI[0][-1]+dI[0]/2, xI[0][0]-dI[0]/2)
    return I, mask, labels, extent


def parallel_lines_2d(thetas, nI, dI):
    """
    Draw a sequence of parallel lines with specified period, thickness , and angle.

    Parameters
    ----------
    thetas: list
        angles at which to draw line patterns.
    nI: tuple of int
    dI: tuple of float
    
    Returns
    -------
    I: numpy array
    mask: numpy array
    labels: numpy array
    extent: tuple

    """
    xI = [(np.arange(n) - (n-1)/2)*d for n,d in zip(nI,dI)]
    xmax = xI[1][-1]
    ymax = xI[0][-1]
    worldtogrid = lambda p,dI,nI : tuple(np.round(x/d + (n-1)/2).astype(int) for x,d,n in zip(p,dI,nI))

    for i in range(len(thetas)):
        m = np.tan(thetas[i])

    pass


def parallel_lines_3d():
    pass



def circle(radius, nI, dI, width=1, noise=0.05, blur=1.0): #, mask_thresh=0.5):
    # create an isotropic image first and downsample later
    # get largest dimension
    maxdim = np.argmax(nI)
    nIiso  = [nI[maxdim]]*len(nI)
    xI = [np.arange(n) - (n-1)//2 for n in nIiso]
    XI = np.stack(np.meshgrid(*xI,indexing='ij'),axis=-1)
    theta = np.arctan(XI[:,::-1,1]/(XI[:,::-1,0]+np.finfo(float).eps))
    I = np.random.randn(*nIiso)*noise
    I_ = np.zeros_like(I)
    for i in range(len(radius)):
        I_ = cv2.circle(I_,(nI[maxdim]//2, nI[maxdim]//2),radius=radius[i], thickness=width, color=(1))
    mask = np.where(I_ > 0.0, 1.0, 0.0)
    labels = theta*mask*(180/np.pi)
    I = I_+I

    # blur = [blur/dI[0],blur/dI[1]]
    I = gaussian_filter(I,sigma=blur)

    # downsample
    I = cv2.resize(I, nI[::-1], interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask,nI[::-1],interpolation=cv2.INTER_NEAREST)
    labels = cv2.resize(labels,nI[::-1],interpolation=cv2.INTER_NEAREST)
    xI = [(np.arange(n) - (n-1)//2)*d for n,d in zip(nI,dI)]
    extent = (xI[1][0]-dI[1]/2, xI[1][-1]+dI[1]/2, xI[0][-1]+dI[0]/2, xI[0][0]-dI[0]/2)

    return I, mask, labels, extent


def ring():
    pass


def anisotropy_correction(image, mask, labels, dI, direction='up', interpolation=cv2.INTER_LINEAR):

    # downsample all dimensions to largest dimension or upsample to the smallest dimension.
    if direction == 'down':
        dim = np.argmax(dI)
    elif direction == 'up':
        dim = np.argmin(dI)
    dsize = [image.shape[dim]]*len(image.shape)
    image_corrected = cv2.resize(image, dsize=dsize, interpolation=interpolation)
    mask_corrected = cv2.resize(mask, dsize=dsize, interpolation=cv2.INTER_NEAREST)
    mask_corrected = np.where(mask_corrected > 0.0, 1.0, 0.0)
    labels_corrected = cv2.resize(labels, dsize=dsize, interpolation=cv2.INTER_NEAREST)
    dI = [dI[dim]]*len(image.shape)
    xI = [(np.arange(n) - (n-1)/2)*d for n,d in zip(image_corrected.shape,dI)]
    extent = (xI[1][0]-dI[1]/2, xI[1][-1]+dI[1]/2, xI[0][-1]+dI[0]/2, xI[0][0]-dI[0]/2) # TODO: generalize for 3D case


    return image_corrected, mask_corrected, labels_corrected, extent