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


def draw_line(image, start_point, end_point, w=1, dI=(1.0,1.0)):
    """Xiaolin Wu's line drawing algorithm.
    
    Parameters
    ----------
    image: numpy array
    start_point: tuple, int
    end_point: tuple, int
    w: float
        line width in the user defined coordinates

    Returns
    -------
    image: np.ndarray
    """
    x0 = start_point[1]
    y0 = start_point[0]
    x1 = end_point[1]
    y1 = end_point[0]

    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        # swap x and y
        x0,y0 = y0,x0
        x1,y1 = y1,x1
        dI = (dI[1], dI[0])
    if x0 > x1:
        # draw backwards
        x0,x1 = x1,x0
        y0,y1 = y1,y0

    dx = x1 - x0
    dy = y1 - y0
    if dx == 0:
        gradient = 1.0
    else:
        gradient = dy/dx

    # Get the vertical component of the line width to find the number of pixels between the top and bottom edge of the line.
    w = w * np.sqrt(1 + (gradient*dI[0]/dI[1])**2) / dI[0] # Here we use the identity 1 + tan^2 = sec^2

    Ix0 = x0
    Ix1 = x1
    y_intercept = y0
    if steep:
        x = Ix0
        while x <= Ix1:
            image[x, int(y_intercept)] = 1-y_intercept%1
            # we want to draw lines with thickness
            # fill pixels with ones between the top and bottom
            for i in range(1,int(w)):
                image[x, int(y_intercept) + i] = 1.0
            image[x, int(y_intercept) + int(w)] = y_intercept%1 # TODO: fix wrap around issue that arises here.
            y_intercept += gradient
            x += 1
    else:
        x = Ix0
        while x <= Ix1:
            image[int(y_intercept), x] = 1-y_intercept%1
            for i in range(1,int(w)):
                image[int(y_intercept) + i, x] = 1.0
            image[int(y_intercept) + int(w), x] = y_intercept%1
            y_intercept += gradient
            x += 1
    return image


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


def get_endpoints(img_borders, theta, p0):
    """
    Find the pixels where the line parameterized by theta and p0 intersects with the image boundaries.

    Parameters
    ----------
    img_borders: tuple
        Max values on each axis in the centered coordinate system
    theta: float
        angle in radians
    p0: tuple
        A point on the line
    
    Returns
    -------
    startpoint: tuple, float
        start point of line (y,x)
    endpoint: tuple, float
        end point of line (y,x)
    out_of_bounds: bool

    """
    slope_intercept_formula = lambda x, slope, b: slope * x + b
    slope = np.tan(theta)
    slope_inv = 1/(slope+np.finfo(float).eps)
    y0, x0 = p0
    ymax, xmax = img_borders
    y_at_xmax = slope_intercept_formula(xmax-x0, slope, y0)
    y_at_xmin = slope_intercept_formula(-xmax-x0, slope, y0)
    x_at_ymax = slope_intercept_formula(ymax-y0, slope_inv, x0)
    x_at_ymin = slope_intercept_formula(-ymax-y0, slope_inv, x0)
    out_of_bounds = np.abs(y_at_xmax) > ymax and np.abs(y_at_xmin) > ymax and np.abs(x_at_ymax) > xmax and np.abs(x_at_ymin) > xmax
    if slope >= 0:
        endpoint = (y_at_xmax, xmax) if np.abs(y_at_xmax) < ymax else (ymax, x_at_ymax)
        startpoint = (y_at_xmin, -xmax) if np.abs(y_at_xmin) < ymax else (-ymax, x_at_ymin)
    else:
        endpoint = (y_at_xmax, xmax) if np.abs(y_at_xmax) < ymax else (-ymax, x_at_ymin)
        startpoint = (y_at_xmin, -xmax) if np.abs(y_at_xmin) < ymax else (ymax, x_at_ymax)

    return startpoint, endpoint, out_of_bounds


def parallel_lines_2d(thetas, nI, dI, width=2, noise=0.1, period=6):
    """
    Draw a sequence of parallel lines with specified period, line width, and angle.

    Parameters
    ----------
    thetas: list
        angles at which to draw line patterns.
    nI: tuple of int
    dI: tuple of float
    width: int
    
    Returns
    -------
    I: numpy array
    labels: numpy array
    extent: tuple

    """
    # create the image with added noise and the mask and labels
    np.random.seed(1)
    I = np.random.randn(*nI)*noise
    labels = np.zeros_like(I)
    # we will need to convert the coordinates with centered origin to array indices
    worldtogrid = lambda p,dI,nI : tuple((x/d + (n-1)/2).astype(int) for x,d,n in zip(p,dI,nI))

    # get the borders of the image
    pad = int(width*np.sqrt(2))
    borders_padded = np.array([((n-1+2*pad)/2)*d for n,d in zip(nI,dI)])
    borders =  np.array([((n-1)/2)*d for n,d in zip(nI,dI)]) # borders are in the defined coordinate system
    # define line endpoints for each field of parallel lines.
    for i in range(len(thetas)):
        I_ = np.zeros([n + 2*pad for n in nI]) #  We need a padded image so line drawing does not go out of borders. It will be cropped at the end.
        theta = thetas[i]
        x_step = np.cos(theta+np.pi/2)*period
        y_step = np.sin(theta+np.pi/2)*period
        cy = 0
        cx = 0
        start, end, _ = get_endpoints(borders_padded, theta, p0=(cy,cx))
        lines = [(start,end)] # list of parallel lines as tuples (startpoint, endpoint)
        while 1:
            cy += y_step
            cx += x_step
            start, end, out_of_bounds = get_endpoints(borders_padded, theta, p0=(cy,cx))
            if out_of_bounds:
                break
            else:
                lines += [(start,end)]
            start, end, out_of_bounds = get_endpoints(borders_padded, theta, p0=(-cy,-cx))
            if out_of_bounds:
                break
            else:
                lines += [(start,end)]

        for j in range(len(lines)):
            start = lines[j][0]
            end = lines[j][1]
            start_point = worldtogrid(start, dI, nI)
            end_point = worldtogrid(end, dI, nI)
            # print(start_point,end_point)
            I_ = draw_line(I_, start_point, end_point, w=width, dI=dI)
        # TODO: this does not give the desired labels. Intersections must be zero or None.
        # add to labels wherever lines don't intersect
        # zero out intersections
        labels_ = theta * (np.where(I_[pad:-pad, pad:-pad],1,0) - np.where(I,1,0))
        labels += np.where(labels_ < 0, 0.0, labels_) # negatives must be reset to zero
        I += I_[pad:-pad, pad:-pad]
    extent = (-borders[1]-dI[1]/2, borders[1]+dI[1]/2, borders[0]+dI[0]/2, -borders[0]-dI[0]/2)
            
    return I, labels, extent

def parallel_lines_3d(thetas, nI, dI, width=2, noise=0.1, period=6):
   """
    Draw a sequence of parallel lines with specified period, line width, and angle.

    Parameters
    ----------
    thetas: list
        angles at which to draw line patterns.
    nI: tuple of int
    dI: tuple of float
    width: int
    
    Returns
    -------
    I: numpy array
    labels: numpy array
    extent: tuple

    """
    # create the image with added noise and the mask and labels
    np.random.seed(1)
    I = np.random.randn(*nI)*noise
    labels = np.zeros_like(I)
    # we will need to convert the coordinates with centered origin to array indices
    worldtogrid = lambda p,dI,nI : tuple((x/d + (n-1)/2).astype(int) for x,d,n in zip(p,dI,nI))

    # get the borders of the image
    pad = int(width*np.sqrt(2))
    borders_padded = np.array([((n-1+2*pad)/2)*d for n,d in zip(nI,dI)])
    borders =  np.array([((n-1)/2)*d for n,d in zip(nI,dI)]) # borders are in the defined coordinate system
    # define line endpoints for each field of parallel lines.
    for i in range(len(thetas)):
        I_ = np.zeros([n + 2*pad for n in nI]) #  We need a padded image so line drawing does not go out of borders. It will be cropped at the end.
        theta = thetas[i]
        x_step = np.cos(theta+np.pi/2)*period
        y_step = np.sin(theta+np.pi/2)*period
        cy = 0
        cx = 0
        start, end, _ = get_endpoints(borders_padded, theta, p0=(cy,cx))
        lines = [(start,end)] # list of parallel lines as tuples (startpoint, endpoint)
        while 1:
            cy += y_step
            cx += x_step
            start, end, out_of_bounds = get_endpoints(borders_padded, theta, p0=(cy,cx))
            if out_of_bounds:
                break
            else:
                lines += [(start,end)]
            start, end, out_of_bounds = get_endpoints(borders_padded, theta, p0=(-cy,-cx))
            if out_of_bounds:
                break
            else:
                lines += [(start,end)]

        for j in range(len(lines)):
            start = lines[j][0]
            end = lines[j][1]
            start_point = worldtogrid(start, dI, nI)
            end_point = worldtogrid(end, dI, nI)
            # print(start_point,end_point)
            I_ = draw_line(I_, start_point, end_point, w=width, dI=dI)
        # add to labels wherever lines don't intersect
        # zero out intersections
        labels_ = theta * (np.where(I_[pad:-pad, pad:-pad],1,0) - np.where(I,1,0))
        labels += np.where(labels_ < 0, 0.0, labels_) # negatives must be reset to zero
        I += I_[pad:-pad, pad:-pad]
    extent = (-borders[1]-dI[1]/2, borders[1]+dI[1]/2, borders[0]+dI[0]/2, -borders[0]-dI[0]/2)
            
    return I, labels, extent

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