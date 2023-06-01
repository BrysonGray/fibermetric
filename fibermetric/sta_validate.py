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
import scipy
import cv2
import pandas as pd
from tqdm.contrib import itertools as tqdm_itertools

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
        gradient = dy/dx  #*dI[0]/dI[1]

    # Get the vertical component of the line width to find the number of pixels between the top and bottom edge of the line.
    # w = w * np.sqrt(1 + (gradient*dI[0]/dI[1])**2) / dI[0] # Here we use the identity 1 + tan^2 = sec^2
    w = w * np.sqrt(1 + gradient**2)/ dI[0]

    Ix0 = x0
    Ix1 = x1
    y_intercept = y0
    if steep:
        x = Ix0
        while x <= Ix1:
            image[x, int(y_intercept)] = 1-y_intercept%1
            # we want to draw lines with thickness
            # fill pixels with ones between the top and bottom
            for i in range(1,int(w)+1):
                image[x, int(y_intercept) + i] = 1.0
            image[x, int(y_intercept) + int(w)+1] = y_intercept%1
            y_intercept += gradient
            x += 1
    else:
        x = Ix0
        while x <= Ix1:
            image[int(y_intercept), x] = 1-y_intercept%1
            for i in range(1,int(w)+1):
                image[int(y_intercept) + i, x] = 1.0
            image[int(y_intercept) + int(w)+1, x] = y_intercept%1
            y_intercept += gradient
            x += 1
    return image


def draw_line_3D(image, start_point, end_point): #, width=1, dI=(1.0,1.0)):
    z1, y1, x1 = start_point
    z2, y2, x2 = end_point
    # ListOfPoints = []
    # ListOfPoints.append((z1, y1, x1))
    image[z1,y1,x1] = 1.0
    dz = abs(z2 - z1)
    dy = abs(y2 - y1)
    dx = abs(x2 - x1)
    if (z2 > z1):
        zs = 1
    else:
        zs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    if (x2 > x1):
        xs = 1
    else:
        xs = -1
 
    # Driving axis is Z-axis"
    if (dz >= dy and dz >= dx):
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while (z1 != z2):
            z1 += zs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dz
            if (p2 >= 0):
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            # ListOfPoints.append((z1, y1, x1))
            image[z1,y1,x1] = 1.0
 
    # Driving axis is Y-axis"
    elif (dy >= dz and dy >= dx):
        p1 = 2 * dz - dy
        p2 = 2 * dx - dy
        while (y1 != y2):
            y1 += ys
            if (p1 >= 0):
                z1 += zs
                p1 -= 2 * dy
            if (p2 >= 0):
                x1 += xs
                p2 -= 2 * dy
            p1 += 2 * dz
            p2 += 2 * dx
            # ListOfPoints.append((z1, y1, x1))
            image[z1,y1,x1] = 1.0
 
    # Driving axis is X-axis"
    else:       
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while (x1 != x2):
            x1 += xs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            # ListOfPoints.append((z1, y1, x1))
            image[z1,y1,x1] = 1.0
    # split the list of points into separate arrays per axis for indexing the image
    # z, y, x = np.array_split(np.array(ListOfPoints), 3, axis=1)
    # image[z,y,x] = 1.0

    return image


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


def parallel_lines_2d(thetas, nI, period=6, width=2, noise=0.1):
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
    dI = [nI[1]/nI[0], 1.0]
    # create the image with added noise and the mask and labels
    np.random.seed(1)
    I = np.random.randn(*nI)*noise
    labels = np.zeros_like(I)
    mask = np.zeros_like(I)
    # we will need to convert the coordinates with centered origin to array indices
    worldtogrid = lambda p,dI,nI : tuple((x/d + (n-1)/2).astype(int) for x,d,n in zip(p,dI,nI))

    # get the borders of the image
    pad = int(width*np.sqrt(2)/min(dI))+1#*np.max(dI)/np.min(dI))
    nI_padded = [n+2*pad for n in nI]
    borders_padded = np.array([((n-1)/2)*d for n,d in zip(nI_padded,dI)])
    borders =  np.array([((n-1)/2)*d for n,d in zip(nI,dI)]) # borders are in the defined coordinate system
    # define line endpoints for each field of parallel lines.
    for i in range(len(thetas)):
        I_ = np.zeros(nI_padded) #  We need a padded image so line drawing does not go out of borders. It will be cropped at the end.
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
            # start_point = worldtogrid(start, dI, nI_padded)
            # end_point = worldtogrid(end, dI, nI_padded)
            I_ = draw_line(I_, start_point, end_point, w=width, dI=dI)
        mask_ = np.where(I_[pad:-pad, pad:-pad],1,0)
        labels += theta * mask_
        mask += mask_
        I += I_[pad:-pad, pad:-pad]
    mask = np.where(mask==1, 1.0, 0.0)
    labels = labels * mask
    extent = (-borders[1]-dI[1]/2, borders[1]+dI[1]/2, borders[0]+dI[0]/2, -borders[0]-dI[0]/2)
            
    return I, labels, extent


def parallel_lines_3D(shape, theta, phi, period, width=1, noise=0.0):
    Z,Y,X = shape
    # any anisotropy is only in the z dimension
    Zhat = Y # The length of Z in units is the same as X and Y. Use Zhat to remember that this is still the Z axis
    zsign = 1 if theta <= np.pi/2 else -1
    ysign = 1 if phi >= 0 else -1
    xsign = 1 if -np.pi/2 <= phi <= np.pi/2 else -1
    # start points scan over two dimensions with the third dimension constant.
    # identify the third dimension. It is the dimension with the fastest changing component along the line.
    # if polar_angle is less than 45 degrees (pi/4), then this is the z dimension
    # else if the azimuthal angle is greater than 45 degrees it is y. Otherwise it is x
    if theta <= np.pi/4 or theta >= 3*np.pi/4:
        # increment x and y
        h = Zhat / np.tan(np.abs(np.pi/2 - theta))
        dz = zsign * Zhat
        dy = h * np.sin(phi)
        dx = h * np.cos(phi)
        zstart = 0 if zsign == 1 else Zhat
        # padz = 0
        ystart = 0 if ysign == 1 else Y + 2*np.ceil(np.abs(dy))
        # pady = int(np.ceil(np.abs(dy)))
        xstart = 0 if xsign == 1 else X + 2*np.ceil(np.abs(dx))
        # padx = int(np.ceil(np.abs(dx)))
        z0 = np.array([zstart])
        y0 = np.arange(ystart, ystart + dy + ysign*Y + period, period)
        x0 = np.arange(xstart, xstart + dx + xsign*X + period, period)

    else: # pi/4 < theta < 3pi/4
        if np.abs(phi) <= np.pi/4 or np.abs(phi) >= 3*np.pi/4:
            # increment y and z
            dz = X * np.tan(np.pi/2 - theta)
            dx = xsign * X
            dy = dx * np.tan(phi)
            zstart = 0 if zsign == 1 else Zhat + np.ceil(2*np.abs(dz))
            # padz = int(np.ceil(np.abs(dz)))
            ystart = 0 if zsign == 1 else Y + np.ceil(2*np.abs(dy))
            # pady = int(np.ceil(np.abs(dy)))
            xstart = 0 if xsign == 1 else X
            # padx = 0
            z0 = np.arange(zstart, zstart + dz + zsign*Zhat + period, period)
            y0 = np.arange(ystart, ystart + dy + ysign*Y + period, period)
            x0 = np.array([xstart])

        else: # pi/4 < abs(phi) < 3pi/4
            # increment x and z
            dz = Y * np.tan(np.pi/2 - theta)
            dy = ysign*Y
            dx = dy/np.tan(phi)
            zstart = 0 if zsign == 1 else Zhat + 2*np.ceil(np.abs(dz))
            # padz = int(np.ceil(np.abs(dz)))
            ystart = 0 if ysign == 1 else Y
            # pady = int(np.ceil(np.abs(dy)))
            xstart = 0 if xsign == 1 else X + 2*np.ceil(np.abs(dx))
            # padx = int(np.ceil(np.abs(dx)))
            z0 = np.arange(zstart, zstart + dz + zsign*Zhat + period, period)
            y0 = np.array([ystart])
            x0 = np.arange(xstart, xstart + dx + xsign*X + period, period)
    z1 = z0 + dz
    y1 = y0 + dy
    x1 = x0 + dx
    start_points = np.round(np.stack(np.meshgrid(z0 * Z/Zhat, y0, x0, indexing='ij'), axis=-1)).astype('int')
    end_points = np.round(np.stack(np.meshgrid(z1 * Z/Zhat, y1, x1, indexing='ij'), axis=-1)).astype('int')
    # initialize a large image with these dimensions to be cropped later.
    large_img_shape = np.max(np.concatenate((start_points,end_points)).reshape(-1,3),axis=0) + 1
    large_img = np.zeros(large_img_shape)
    pad = (large_img_shape - shape)//2
    r = (large_img_shape - shape)%2
    # draw lines in the large image
    for start,end in zip(start_points.reshape(-1,3), end_points.reshape(-1,3)):
        large_img = draw_line_3D(large_img, start, end)
    # crop the large image to the final shape.

    img = large_img[pad[0]:large_img_shape[0]-(pad[0]+r[0]), pad[1]:large_img_shape[1]-(pad[1]+r[1]), pad[2]:large_img_shape[2]-(pad[2]+r[2])]
    print('test')
    # binary dilation to add thickness to lines
    # first in the xy plane

    # k = 0
    # for i in range(width):
    #     if ((i+1) * Z/Y)//1 - k == 1:
    #         img = scipy.ndimage.binary_dilation(img, structure=np.ones((3,3,3)))
    #         k += 1
    #     else:
    #         img = scipy.ndimage.binary_dilation(img, structure=np.ones((1,3,3)))
    # img = img.astype('float')
    # blur the image for anti-aliasing
    # img = gaussian_filter(img, sigma=(Z/Y,1,1))

    return img


def downsample_image(image, downsample_factor):
    """
    Downsample a given image array by taking the local average of each block of pixels 
    with size downsample_factor x downsample_factor.
    
    Args:
        image (ndarray): a 2D numpy array representing the image.
        downsample_factor (int): the size of the block of pixels for local averaging.
        
    Returns:
        The downsampled image array.
    """
    m, n = image.shape
    if isinstance(downsample_factor, (tuple, list, np.ndarray)):
        down_m = int(downsample_factor[0])
        down_n = int(downsample_factor[1])
    else:
        down_m = down_n = downsample_factor
    # Compute the number of blocks of pixels along each axis.
    block_m = m//down_m
    block_n = n//down_n
    
    # Reshape the image array into a 4D array of shape (block_m, downsample_factor, block_n, downsample_factor), 
    # with each element representing a block of pixels.
    blocks = image[:block_m*down_m, :block_n*down_n].reshape(block_m, down_m, block_n, down_n)
    
    # Take the mean of each block of pixels along the width and height dimensions.
    downsampled_blocks = blocks.mean((1, 3))
    
    # Reshape the downsampled blocks into a 2D numpy array and return.
    return downsampled_blocks.reshape(block_m, block_n)

def circle(nI, period=6, width=1, noise=0.05, blur=0.0, min_radius=4, ): #, mask_thresh=0.5):
    # create an isotropic image first and downsample later
    # get largest dimension
    dI = [nI[1]/nI[0], 1.0]
    maxdim = np.argmax(nI)
    nIiso  = [nI[maxdim]]*len(nI)
    xI = [np.arange(n) - (n-1)//2 for n in nIiso]
    XI = np.stack(np.meshgrid(*xI,indexing='ij'),axis=-1)
    # theta = np.arctan(XI[:,::-1,1]/(XI[:,::-1,0]+np.finfo(float).eps))
    theta = np.arctan2(XI[:,::-1,1], XI[:,::-1,0])
    I = np.random.randn(*nIiso)*noise
    I_ = np.zeros_like(I)
    max_radius = np.sqrt(nI[1]**2/2)
    radii = np.arange(min_radius, max_radius, period, dtype=int)
    for i in range(len(radii)):
        I_ = cv2.circle(I_,(nI[maxdim]//2, nI[maxdim]//2),radius=radii[i], thickness=width, color=(1))
    # mask = np.where(I_ > 0.0, 1.0, 0.0)
    labels = theta #*mask #*(180/np.pi)
    I = I_+I

    # blur = [blur/dI[0],blur/dI[1]]
    I = gaussian_filter(I,sigma=blur)

    # downsample
    # I = cv2.resize(I, nI[::-1], interpolation=cv2.INTER_AREA)
    I = downsample_image(I, (nI[1]/nI[0], 1))
    # mask = cv2.resize(mask,nI[::-1],interpolation=cv2.INTER_NEAREST)
    # labels = cv2.resize(labels,nI[::-1],interpolation=cv2.INTER_NEAREST)
    labels = downsample_image(labels, (nI[1]/nI[0], 1))
    xI = [(np.arange(n) - (n-1)//2)*d for n,d in zip(nI,dI)]
    extent = (xI[1][0]-dI[1]/2, xI[1][-1]+dI[1]/2, xI[0][-1]+dI[0]/2, xI[0][0]-dI[0]/2)

    return I, labels, extent


def ring():
    pass


def anisotropy_correction(image, labels, dI, direction='up', interpolation=cv2.INTER_AREA):

    # downsample all dimensions to largest dimension or upsample to the smallest dimension.
    if direction == 'down':
        dim = np.argmax(dI)
    elif direction == 'up':
        dim = np.argmin(dI)
    dsize = [image.shape[dim]]*len(image.shape)
    image_corrected = cv2.resize(image, dsize=dsize, interpolation=interpolation)
    # mask_corrected = cv2.resize(mask, dsize=dsize, interpolation=cv2.INTER_NEAREST)
    # mask_corrected = np.where(mask_corrected > 0.0, 1.0, 0.0)
    thetas = np.unique(labels)
    labels_corrected = cv2.resize(labels, dsize=dsize, interpolation=interpolation)
    # set all values in labels_corrected to the nearest value in thetas
    label_ids = np.abs(labels_corrected[:, :, None] - thetas[None, None, :]).argmin(axis=2)
    labels_corrected = thetas[label_ids]
    dI = [dI[dim]]*len(image.shape)
    xI = [(np.arange(n) - (n-1)/2)*d for n,d in zip(image_corrected.shape,dI)]
    extent = (xI[1][0]-dI[1]/2, xI[1][-1]+dI[1]/2, xI[0][-1]+dI[0]/2, xI[0][0]-dI[0]/2) # TODO: generalize for 3D case


    return image_corrected, labels_corrected, extent


def phantom_test(derivative_sigma, tensor_sigma, nI, period=6, width=1, noise=0.05, phantom='grid', err_type='pixelwise', grid_thetas=None, tile_size=None, dim=2, display=False):
    """Test structure tensor analysis on a grid of crossing lines.
    Parameters
    ----------
    dim : int
        Number of dimensions.
    derivative_sigma : list, float
        Sigma for the derivative filter.
    tensor_sigma : float
        Sigma for the structure tensor filter.
    nI : tuple of int
        Number of pixels in each dimension.
    grid_thetas : tuple of float
        Angles of the lines in radians. One angle for 2D, two angles (polar and azimuthal) for 3D.
    width : int
        Width of the lines.
    noise : float
        Noise level.
    blur : float
        Blur level.
    period : int
        Space between lines.
    phantom : ['grid', 'circles']
        Phantom type
    err_type : ['pixelwise', 'piecewise']
        Pixelwise returns average angular difference per pixel. 
        Piecewise divides the image into pieces and fits each group of pixels to an ODF from which peaks are computed and compared to ground truth.
    
    Returns
    -------
    error : float
        Average angular difference between ground truth and estimated angles for pixelwise error, or jensen-shannon divergence for piecewise error.

    """
    if dim == 2:
        dI = (nI[1]/nI[0], 1.0)
        assert phantom in ('grid', 'circles')
        if phantom == 'grid':
            assert isinstance(grid_thetas, (list, tuple)), 'grid_thetas must be a list or tuple of angles when using a grid phantom.'
            I, labels, extent = parallel_lines_2d(grid_thetas, nI, width=width, noise=noise, period=period)
        elif phantom == 'circles':
            I, labels, extent = circle(nI, period=period, width=width, noise=noise)
        # apply anisotropy correction
        I, labels, extent = anisotropy_correction(I, labels, dI)
        # compute structure tensor and angles
        S = histology.structure_tensor(I, derivative_sigma=derivative_sigma, tensor_sigma=tensor_sigma, dI=(1.0,1.0))
        angles = histology.angles(S)[0]
        assert err_type in ('pixelwise', 'piecewise')
        if err_type == 'pixelwise':
            # count number of angles that are not none
            nangles = np.sum(~np.isnan(angles))
            # compute the average difference between computed angles and labels
            # get angles where not none
            # angles_ = angles[~np.isnan(angles)]
            # labels_ = labels[~np.isnan(angles)]
            angles_ = angles
            angles_flipped = np.where(angles_ < 0, angles_ + np.pi, angles_ - np.pi)
            angles_ = np.stack((angles_, angles_flipped), axis=-1)
            labels_ = labels
            labels_flipped = np.where(labels_ < 0, labels_ + np.pi, labels_ - np.pi)
            labels_ = np.stack((labels_, labels_flipped), axis=-1)
            diff1 = np.abs(angles_ - labels_)
            diff2 = np.abs(angles_[...,::-1] - labels_)
            diff = np.concatenate((diff1, diff2), axis=-1)
            diff = np.nanmin(diff, axis=-1)
            error = np.nanmean(diff) * 180 / np.pi # average error in radians
        elif err_type == 'piecewise':
            if tile_size is None:
                tile_size = nI[1] // 10 # default to ~100 tiles in the image
            # odf, sample_points = histology.odf2d(angles, nbins=100, tile_size=tile_size)
            odf, mu, kappa, pi = histology.odf2d_vonmises(angles, nbins=100, tile_size=tile_size)
            grid_thetas = np.sort(np.array(grid_thetas))
            mu = np.sort(mu, axis=-1)
            error = np.mean(np.abs(mu - grid_thetas[None,None]))*180/np.pi # TODO: something may be wrong here

            # # compare odf to the ground truth distribution using jenson-shannon divergence.
            # # the ground truth is a distribution of delta functions at the angles of the lines.
            # thetas_symmetric = np.concatenate((grid_thetas, np.array(grid_thetas) + np.pi))
            # thetas_symmetric = np.where(thetas_symmetric > np.pi, thetas_symmetric - 2*np.pi, thetas_symmetric)
            # ground_truth = np.zeros(odf.shape[-1])
            # ground_truth[np.digitize(thetas_symmetric, sample_points)] = 1.0 / len(thetas_symmetric)
            # # ground_truth = gaussian_filter(ground_truth, sigma=1)
            # js = np.apply_along_axis(lambda a: scipy.spatial.distance.jensenshannon(a, ground_truth), axis=-1, arr=odf)
            # error = np.mean(js)
        
        if display:
            fig, ax = plt.subplots(1,4, figsize=(12,4))
            ax[0].imshow(I, extent=extent)
            ax[0].set_title('Image')
            ax[1].imshow(labels, extent=extent)
            ax[1].set_title('Ground Truth')
            ax[2].imshow(angles, extent=extent)
            ax[2].set_title('Angles')
            ax[3].imshow(diff, extent=extent)
            ax[3].set_title('Difference')
            plt.show()

            # # The below method produces errors because scipy.signal.find_peaks does not always find the exact number of peaks as thetas.
            # odf_peaks = np.apply_along_axis(lambda a: scipy.signal.find_peaks(a, prominence=0.005)[0], axis=-1, arr=odf)
            # # take only angles in the range -pi/2, pi/2
            # peaks = np.apply_along_axis(lambda a: [p for p in sample_points[a] if p >= -np.pi/2 and p <= np.pi/2 ], axis=-1, arr=odf_peaks)
            # assert peaks.shape[-1] == len(thetas), 'number of peaks detected does not equal the number of angles in the image'
            # thetas.sort() # peaks are already be in order from least to greatest
            # error.append( np.sum(np.abs(peaks - thetas)) / peaks.shape )

    return error


def run_tests(derivative_sigmas, tensor_sigmas, nIs, periods=[6], widths=[1], noises=[0.05], phantom='grid', err_type='pixelwise', grid_thetas=[None], tile_size=None, dim=2):
    error_df = pd.DataFrame({'derivative_sigma':[], 'tensor_sigma':[], 'nI':[], 'period':[], 'width':[], 'noise':[], 'phantom':[], 'error type':[], 'grid thetas':[], 'tile size':[], 'dimensions':[], 'error':[]})
    # ensure all arguments are lists
    if not isinstance(derivative_sigmas, (list, tuple, np.ndarray)):
        derivative_sigmas = [derivative_sigmas]
    if not isinstance(tensor_sigmas, (list, tuple, np.ndarray)):
        tensor_sigmas = [tensor_sigmas]
    if not isinstance(nIs[0], (list, tuple, np.ndarray)):
        nIs = [nIs]
    if not isinstance(periods, (list, tuple, np.ndarray)):
        periods = [periods]
    if not isinstance(grid_thetas[0], (list, tuple, np.ndarray)):
        grid_thetas = [grid_thetas]
    if not isinstance(widths, (list, tuple, np.ndarray)):
        widths = [widths]
    if not isinstance(noises, (list, tuple, np.ndarray)):
        noises = [noises]
    
    for i1,i2,i3,i4,i5,i6,i7 in tqdm_itertools.product(range(len(derivative_sigmas)), range(len(tensor_sigmas)), range(len(nIs)), range(len(periods)), range(len(widths)), range(len(noises)), range(len(grid_thetas))):
        derivative_sigma = derivative_sigmas[i1]
        tensor_sigma = tensor_sigmas[i2]
        nI = nIs[i3]
        period = periods[i4]
        width = widths[i5]
        noise = noises[i6]
        thetas = grid_thetas[i7]
        error = phantom_test(derivative_sigma, tensor_sigma, nI, period, width, noise, phantom, err_type, thetas, tile_size, dim)
        new_row = {'derivative_sigma': derivative_sigma, 'tensor_sigma': tensor_sigma, 'nI': [nI], 'period': period, 'width': width, 'noise': noise, 'phantom': phantom, 'error type': err_type, 'grid thetas': [thetas], 'tile size': tile_size, 'dimensions': dim, 'error': error}
        error_df = pd.concat([error_df, pd.DataFrame(new_row)], ignore_index=True)
    return error_df

    


# scratch
############################################################################################################################################################################

# # if some arguments are a single value, make them a list of length equal to len(derivative_sigma)
# nI = _make_list(nI, len(derivative_sigma))
# period = _make_list(period, len(derivative_sigma))
# width = _make_list(width, len(derivative_sigma))
# noise = _make_list(noise, len(derivative_sigma))
# if thetas is not None:
#     thetas = _make_list(thetas, len(derivative_sigma))

# def _make_list(a, length):
#     if not isinstance(a, list):
#         a = [a]*length
#     elif len(a) == 1:
#         a = a*length
#     return a

# Python3 code for generating points on a 3-D line
# using Bresenham's Algorithm

# def synthetic2d(thetas: tuple, nI: tuple[int], dI: tuple[float], width: int= 2, noise: float=0.1, blur=1.0, mask_thresh: float=0.1):
#     xI = [(np.arange(n) - (n-1)/2)*d for n,d in zip(nI,dI)]
#     xmax = xI[1][-1]
#     ymax = xI[0][-1]
#     np.random.seed(1)
#     I = np.random.randn(*nI)*noise
#     mask = np.zeros_like(I)
#     labels = np.zeros_like(I)
#     worldtogrid = lambda p,dI,nI : tuple(np.round(x/d + (n-1)/2).astype(int) for x,d,n in zip(p,dI,nI))
#     for i in range(len(thetas)):
#         theta = thetas[i]
#         m = np.tan(theta)
#         y0 = m*xmax
#         y1 = -y0
#         x0 = ymax/(m+np.finfo(float).eps)
#         x1 = -x0
#         if np.abs(x0) > xmax:
#             x0 = xmax
#             x1 = -xmax
#         elif np.abs(y0) > ymax:
#             y0 = ymax
#             y1 = -ymax
#         j0,i0 = worldtogrid((y0,x0),dI,nI)
#         j1,i1 = worldtogrid((y1,x1),dI,nI)
#         start_point = (i0,j0)
#         end_point = (i1,j1)
#         I_ = cv2.line(np.zeros_like(I),start_point,end_point,thickness=width, color=(1))
#         mask += I_
#         labels += np.where(I_==1.0, theta*(180/np.pi), 0.0)
#         I += I_
#     mask = np.where(mask==1.0, mask, 0.0)
#     labels = labels*mask
#     I = np.where(I > 1.2, 1.0, I)
#     blur = [blur/dI[0],blur/dI[1]]
#     I = gaussian_filter(I,sigma=blur)
#     extent = (xI[1][0]-dI[1]/2, xI[1][-1]+dI[1]/2, xI[0][-1]+dI[0]/2, xI[0][0]-dI[0]/2)
#     return I, mask, labels, extent

# def synthetic_circle(radius, nI, dI, width=1, noise=0.05, blur=1.0): #, mask_thresh=0.5):
#     # create an isotropic image first and downsample later
#     # get largest dimension
#     maxdim = np.argmax(nI)
#     nIiso  = [nI[maxdim]]*len(nI)
#     xI = [np.arange(n) - (n-1)//2 for n in nIiso]
#     XI = np.stack(np.meshgrid(*xI,indexing='ij'),axis=-1)
#     theta = np.arctan(XI[:,::-1,1]/(XI[:,::-1,0]+np.finfo(float).eps))
#     I = np.random.randn(*nIiso)*noise
#     I_ = np.zeros_like(I)
#     for i in range(len(radius)):
#         I_ = cv2.circle(I_,(nI[maxdim]//2, nI[maxdim]//2),radius=radius[i], thickness=width, color=(1))
#     mask = np.where(I_ > 0.0, 1.0, 0.0)
#     labels = theta*mask*(180/np.pi)
#     I = I_+I

#     # blur = [blur/dI[0],blur/dI[1]]
#     I = gaussian_filter(I,sigma=blur)

#     # downsample
#     I = cv2.resize(I, nI[::-1], interpolation=cv2.INTER_AREA)
#     mask = cv2.resize(mask,nI[::-1],interpolation=cv2.INTER_NEAREST)
#     labels = cv2.resize(labels,nI[::-1],interpolation=cv2.INTER_NEAREST)
#     xI = [(np.arange(n) - (n-1)//2)*d for n,d in zip(nI,dI)]
#     extent = (xI[1][0]-dI[1]/2, xI[1][-1]+dI[1]/2, xI[0][-1]+dI[0]/2, xI[0][0]-dI[0]/2)

#     return I, mask, labels, extent

# def synthetic3d():
#     pass


# def anisotropy_correction(image, mask, labels, dI, direction='down', interpolation=cv2.INTER_LINEAR):

#     # downsample all dimensions to largest dimension or upsample to the smallest dimension.
#     if direction == 'down':
#         dim = np.argmax(dI)
#     elif direction == 'up':
#         dim = np.argmin(dI)
#     dsize = [image.shape[dim]]*len(image.shape)
#     image_corrected = cv2.resize(image, dsize=dsize, interpolation=interpolation)
#     mask_corrected = cv2.resize(mask, dsize=dsize, interpolation=cv2.INTER_NEAREST)
#     mask_corrected = np.where(mask_corrected > 0.0, 1.0, 0.0)
#     labels_corrected = cv2.resize(labels, dsize=dsize, interpolation=cv2.INTER_NEAREST)
#     dI = [dI[dim]]*len(image.shape)
#     xI = [(np.arange(n) - (n-1)/2)*d for n,d in zip(image_corrected.shape,dI)]
#     extent = (xI[1][0]-dI[1]/2, xI[1][-1]+dI[1]/2, xI[0][-1]+dI[0]/2, xI[0][0]-dI[0]/2) # TODO: generalize for 3D case


#     return image_corrected, mask_corrected, labels_corrected, extent

# FROM: https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/
 
# def Bresenham3D(x1, y1, z1, x2, y2, z2):
#     ListOfPoints = []
#     ListOfPoints.append((x1, y1, z1))
#     dx = abs(x2 - x1)
#     dy = abs(y2 - y1)
#     dz = abs(z2 - z1)
#     if (x2 > x1):
#         xs = 1
#     else:
#         xs = -1
#     if (y2 > y1):
#         ys = 1
#     else:
#         ys = -1
#     if (z2 > z1):
#         zs = 1
#     else:
#         zs = -1
 
#     # Driving axis is X-axis"
#     if (dx >= dy and dx >= dz):       
#         p1 = 2 * dy - dx
#         p2 = 2 * dz - dx
#         while (x1 != x2):
#             x1 += xs
#             if (p1 >= 0):
#                 y1 += ys
#                 p1 -= 2 * dx
#             if (p2 >= 0):
#                 z1 += zs
#                 p2 -= 2 * dx
#             p1 += 2 * dy
#             p2 += 2 * dz
#             ListOfPoints.append((x1, y1, z1))
 
#     # Driving axis is Y-axis"
#     elif (dy >= dx and dy >= dz):      
#         p1 = 2 * dx - dy
#         p2 = 2 * dz - dy
#         while (y1 != y2):
#             y1 += ys
#             if (p1 >= 0):
#                 x1 += xs
#                 p1 -= 2 * dy
#             if (p2 >= 0):
#                 z1 += zs
#                 p2 -= 2 * dy
#             p1 += 2 * dx
#             p2 += 2 * dz
#             ListOfPoints.append((x1, y1, z1))
 
#     # Driving axis is Z-axis"
#     else:       
#         p1 = 2 * dy - dz
#         p2 = 2 * dx - dz
#         while (z1 != z2):
#             z1 += zs
#             if (p1 >= 0):
#                 y1 += ys
#                 p1 -= 2 * dz
#             if (p2 >= 0):
#                 x1 += xs
#                 p2 -= 2 * dz
#             p1 += 2 * dy
#             p2 += 2 * dx
#             ListOfPoints.append((x1, y1, z1))
#     return ListOfPoints
 
 
# def main():
#     (x1, y1, z1) = (-1, 1, 1)
#     (x2, y2, z2) = (5, 3, -1)
#     ListOfPoints = Bresenham3D(x1, y1, z1, x2, y2, z2)
#     print(ListOfPoints)
 
# main()