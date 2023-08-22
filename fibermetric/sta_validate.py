#!/usr/bin/env python
# ruff: noqa: E501
'''
Structure tensor analysis validation functions.

Author: Bryson Gray
2023

'''

import sys
from tkinter import N
sys.path.insert(0, '/home/brysongray/fibermetric/')
sys.path.insert(0, '/home/brysongray/periodic-kmeans/')
from periodic_kmeans.periodic_kmeans import periodic_kmeans, PeriodicKMeans
from fibermetric import histology, apsym_kmeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,sobel,correlate1d,gaussian_filter1d
import scipy
import cv2
from torch.nn import Upsample
import torch
import pandas as pd
from tqdm.contrib import itertools as tqdm_itertools
from sklearn.cluster import KMeans

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
    w = w * np.sqrt(1 + gradient**2)/ dI[0] # Here we use the identity 1 + tan^2 = sec^2 and divide by dI to convert from user defined coordinates to pixels

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


# def draw_line_3D(image, start_point, end_point): #, width=1, dI=(1.0,1.0)):
#     z1, y1, x1 = start_point
#     z2, y2, x2 = end_point
#     # ListOfPoints = []
#     # ListOfPoints.append((z1, y1, x1))
#     image[z1,y1,x1] = 1.0
#     dz = abs(z2 - z1)
#     dy = abs(y2 - y1)
#     dx = abs(x2 - x1)
#     if (z2 > z1):
#         zs = 1
#     else:
#         zs = -1
#     if (y2 > y1):
#         ys = 1
#     else:
#         ys = -1
#     if (x2 > x1):
#         xs = 1
#     else:
#         xs = -1
 
#     # Driving axis is Z-axis"
#     if (dz >= dy and dz >= dx):
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
#             # ListOfPoints.append((z1, y1, x1))
#             image[z1,y1,x1] = 1.0
 
#     # Driving axis is Y-axis"
#     elif (dy >= dz and dy >= dx):
#         p1 = 2 * dz - dy
#         p2 = 2 * dx - dy
#         while (y1 != y2):
#             y1 += ys
#             if (p1 >= 0):
#                 z1 += zs
#                 p1 -= 2 * dy
#             if (p2 >= 0):
#                 x1 += xs
#                 p2 -= 2 * dy
#             p1 += 2 * dz
#             p2 += 2 * dx
#             # ListOfPoints.append((z1, y1, x1))
#             image[z1,y1,x1] = 1.0
 
#     # Driving axis is X-axis"
#     else:       
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
#             # ListOfPoints.append((z1, y1, x1))
#             image[z1,y1,x1] = 1.0
#     # split the list of points into separate arrays per axis for indexing the image
#     # z, y, x = np.array_split(np.array(ListOfPoints), 3, axis=1)
#     # image[z,y,x] = 1.0

#     return image

def draw_line_3D(I, XI, direction, point, inv_sigma, norm, labels=None, line_thresh=None, display=False, angles=None):
    projection = (XI - point) - ((XI - point)@direction.T)*direction
    xy_dist = np.linalg.norm(projection[...,1:], axis=-1)
    z_dist = projection[...,0]
    dist = np.stack([z_dist,xy_dist],-1)[...,None,:]
    line = np.exp(-0.5 * dist @ inv_sigma @ dist.transpose(0,1,2,4,3)).squeeze() / norm
    I += line
    if display:
        line_label = np.where(line[...,None] > line_thresh, angles, [0.0,0.0])
        labels += line_label

    return I, labels


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
    slope = np.tan(theta)
    # y,x = p0
    # x0 = x - slope*y
    # a = np.array([-np.sin(theta), np.cos(theta)]) # the vector normal to the line
    # # vertices = np.array([[-y0,0], [-y0,img_borders[1]], [img_borders[0]-y0,img_borders[1]], [img_borders[0]-y0,0]])
    # vertices = np.array([[0,-x0], [0,img_borders[1]-x0], [img_borders[0],img_borders[1]-x0], [img_borders[0],-x0]])
    # # project the vertices onto the normal vector
    # proj = vertices@a[:,None]
    # # if the projection of the vertices onto the normal vector are all positive or all negative, then the line does not intersect the polygon
    # out_of_bounds = np.all(proj > 0) or np.all(proj < 0)
    def slope_intercept_formula(x, slope, b):
        return slope * x + b
    y0, x0 = p0
    slope_inv = 1/(slope+np.finfo(float).eps)
    ymax, xmax = img_borders
    y_at_xmax = slope_intercept_formula(xmax-x0, slope, y0)
    y_at_xmin = slope_intercept_formula(-xmax-x0, slope, y0)
    x_at_ymax = slope_intercept_formula(ymax-y0, slope_inv, x0)
    x_at_ymin = slope_intercept_formula(-ymax-y0, slope_inv, x0)
    # y_at_xmax = slope_inv*(xmax-x) + y
    # y_at_xmin = slope_inv*(-xmax-x) + y
    # x_at_ymax = slope*(ymax-y) + x
    # x_at_ymin = slope*(-ymax-y) + x
    out_of_bounds = np.abs(y_at_xmax) > ymax and np.abs(y_at_xmin) > ymax and np.abs(x_at_ymax) > xmax and np.abs(x_at_ymin) > xmax
    if slope >= 0:
        endpoint = (y_at_xmax, xmax) if np.abs(y_at_xmax) <= ymax else (ymax, x_at_ymax)
        startpoint = (y_at_xmin, -xmax) if np.abs(y_at_xmin) <= ymax else (-ymax, x_at_ymin)
    else:
        endpoint = (y_at_xmax, xmax) if np.abs(y_at_xmax) <= ymax else (-ymax, x_at_ymin)
        startpoint = (y_at_xmin, -xmax) if np.abs(y_at_xmin) <= ymax else (ymax, x_at_ymax)

    return startpoint, endpoint, out_of_bounds


def radial_lines_2d(thetas: tuple, nI: "tuple[int]", dI: "tuple[float]", width: int= 2, noise: float=0.1, blur=1.0, mask_thresh: float=0.1):

    xI = [(np.arange(n) - (n-1)/2)*d for n,d in zip(nI,dI)]
    xmax = xI[1][-1]
    ymax = xI[0][-1]
    np.random.seed(1)
    I = np.random.randn(*nI)*noise
    mask = np.zeros_like(I)
    labels = np.zeros_like(I)
    def worldtogrid(p, dI, nI):
        return tuple(np.round(x / d + (n - 1) / 2).astype(int) for x, d, n in zip(p, dI, nI))
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
    def worldtogrid(p, dI, nI):
        return tuple((x / d + (n - 1) / 2).astype(int) for x, d, n in zip(p, dI, nI))

    # get the borders of the image
    # pad the image to account for the line width and anti-aliasing. 
    pad = int(width*np.sqrt(2)/min(dI))+1 # this is the max width of the line in pixels plus 1. Refer to draw_line for width calculation details.
    # borders are in the defined coordinate system. Add padding because we'll crop the left side to cut off the flat end of the line.
    borders = np.array([((n-1)/2)*d for n,d in zip([n+pad for n in nI],dI)])
    # define line endpoints for each field of parallel lines.
    for i in range(len(thetas)):
        # We need a padded image so line drawing does not go out of borders. It will be cropped at the end.
        # Add 2*pad because we will crop the left and the right side of the image.
        I_ = np.zeros([n+2*pad for n in nI])
        theta = thetas[i]
        x_step = np.cos(theta+np.pi/2)*period
        y_step = np.sin(theta+np.pi/2)*period
        # x_step = np.cos(theta)*period
        # y_step = np.sin(theta)*period
        cy = 0
        cx = 0
        start, end, _ = get_endpoints(borders, theta, p0=(cy,cx))
        lines = [(start,end)] # list of parallel lines as tuples (startpoint, endpoint)
        while 1:
            cy += y_step
            cx += x_step

            start, end, out_of_bounds = get_endpoints(borders, theta, p0=(cy,cx))
            if out_of_bounds:
                break
            else:
                lines += [(start,end)]
            start, end, out_of_bounds = get_endpoints(borders, theta, p0=(-cy,-cx))
            if out_of_bounds:
                break
            else:
                lines += [(start,end)]

        for j in range(len(lines)):
            start = lines[j][0]
            end = lines[j][1]
            start_point = worldtogrid(start, dI, [n+pad for n in nI])
            end_point = worldtogrid(end, dI, [n+pad for n in nI])
            I_ = draw_line(I_, start_point, end_point, w=width, dI=dI)
        mask_ = np.where(I_[pad:-pad, pad:-pad],1,0)
        labels += theta * mask_
        mask += mask_
        I += I_[pad:-pad, pad:-pad]
    mask = np.where(mask==1, 1.0, 0.0)
    labels = labels * mask
    extent = (0,nI[1]*dI[1], nI[0]*dI[0], 0)
            
    return I, labels, extent


def out_of_bounds(img_borders, angle, p):
    """ determine if the line intersects with the convex polygon defined by the image borders.
    """
    if len(img_borders) == 2:
        # enforce theta is a scalar or a 1D array
        if not isinstance(angle, (int,float)):
            raise ValueError('angle must be a scalar for 2D images')
        a = np.array([-np.sin(angle), np.cos(angle)]) # the vector normal to the line
        vertices = np.array([[0.,0.],[0.,img_borders[1]],[img_borders[0],img_borders[1]],[img_borders[0],0.]])
        vertices = vertices - np.array(p)
        # project the vertices onto the normal vector
        proj = vertices@a[:,None]
        # if the projection of the vertices onto the normal vector are all positive or all negative, then the line does not intersect the polygon
        out_of_bounds = np.all(proj > 0) or np.all(proj < 0)
    elif len(img_borders) == 3:
        if not hasattr(angle, '__len__'):
            raise ValueError('angle must be a list or tuple of two values (theta,phi) for 3D images.')
        elif len(angle) != 2:
            raise ValueError('angle must be a list or tuple of two values (theta,phi) for 3D images.')
        # theta, phi = angle
        # direction_vector = np.array([np.cos(theta), np.sin(theta)*np.sin(phi), np.sin(theta)*np.cos(phi)])
        # zmax,ymax,xmax = img_borders
        # vertices = np.array([[0.,0.,0.],[0.,ymax,0.],[0.,ymax,xmax],[0.,0.,xmax],
        #                      [zmax,0.,0.],[zmax,ymax,0.],[zmax,ymax,xmax],[zmax,0.,xmax]])
        # vertices = vertices - np.array(p)
        # cp = np.cross(vertices, direction_vector)
        # # check if any of the cross products are in the opposite octant of any other (do the vectors exist in a half-space?)
        # mean_cp = np.mean(cp, axis=1)
        # out_of_bounds = np.any(np.all(cp*mean_cp[:,None] < 0, axis=0))

        theta, phi = angle
        if theta <= np.pi/4 or theta >= 3*np.pi/4:
            # incrementing x and y
            boundaries = np.array([0., np.cos(phi)*np.tan(theta)*img_borders[0]/2, np.sin(phi)*np.tan(theta)*img_borders[0]/2])
            out_of_bounds = np.any(np.abs(p) > boundaries+img_borders)
        elif np.abs(phi) <= np.pi/4 or np.abs(phi) >= 3*np.pi/4:
            # incrementing x and z
            boundaries = np.array([img_borders[1]/2/np.cos(phi)*np.tan(theta), 0., img_borders[1]/2*np.tan(phi)])
            out_of_bounds = np.any(np.abs(p) > boundaries+img_borders)
        else: # pi/4 < abs(phi) < 3pi/4
            # incrementing y and z
            boundaries = np.array([img_borders[1]/2/np.cos(phi)*np.tan(theta), img_borders[2]/2/np.tan(phi), 0.])
            out_of_bounds = np.any(np.abs(p) > boundaries+img_borders)

    return out_of_bounds

def parallel_lines_2d_v01(thetas, nI, period=6, noise=0.1, display=False):
    multiple = 1.0
    dI = [nI[1]/nI[0], 1.0]
    xI = [np.arange(n)*d for n,d in zip(nI,dI)]
    XI = np.stack(np.meshgrid(*xI, indexing='ij'), axis=-1)
    # create the image with added noise and the mask and labels
    np.random.seed(1)
    I = np.random.randn(*nI)*noise
    labels = np.zeros(nI)

    for i in range(len(thetas)):
        theta = thetas[i]
        y_step = -np.sin(theta)*period
        x_step = np.cos(theta)*period

        sigma = (np.sin(theta)*dI[0]*multiple)**2 + (np.cos(theta)*dI[1]*multiple)**2
        line_thresh = np.exp(-0.5)/np.sqrt(2.0*np.pi*sigma)

        # draw the line through the center of the image
        c0 = np.array([(nI[0]-1)*dI[0]/2, (nI[1]-1)*dI[1]/2])
        # dist = (XI - c0)@np.array([-np.sin(theta),np.cos(theta)])
        dist = (XI - c0)@np.array([-np.sin(theta), np.cos(theta)])
        line = np.exp(-0.5 * dist**2 / sigma) / np.sqrt(2.0*np.pi*sigma)
        I += line
        if display == True:
            line_label = np.where(line > line_thresh, theta, 0.0)
            labels += line_label

        shift = np.array([0.0, 0.0])
        while 1:
            shift += np.array([y_step, x_step])
            # shift the line to the left and right by the period
            center = c0 + shift
            # check if the line is out of bounds
            oob = out_of_bounds(((nI[0]-1)*dI[0], (nI[1]-1)*dI[1]), theta, center)
            if oob:
                break
            dist = (XI - center)@np.array([-np.sin(theta), np.cos(theta)])
            line = np.exp(-0.5 * dist**2 / sigma) /np.sqrt(2.0*np.pi*sigma)
            I += line
            if display == True:
                line_label = np.where(line > line_thresh, theta, 0.0)
                labels += line_label


            center = c0 - shift
            oob = out_of_bounds(((nI[0]-1)*dI[0], (nI[1]-1)*dI[1]), theta, center)
            if oob:
                break
            # dist = (XI - center)@np.array([-np.sin(theta),np.cos(theta)])
            dist = (XI - center)@np.array([-np.sin(theta), np.cos(theta)])
            line = np.exp(-0.5 * dist**2 / sigma) /np.sqrt(2.0*np.pi*sigma)
            I += line
            if display == True:
                line_label = np.where(line > line_thresh, theta, 0.0)
                labels += line_label

    labels = np.where(np.any(labels[...,None] == np.array(thetas)[None,None], axis=2), labels, 0.0)

    extent = (0,nI[1]*dI[1], nI[0]*dI[0], 0)

    return I, labels, extent


def parallel_lines_3D(shape, theta, phi, period, width=1, noise=0.0):
    """
    shape: tuple
    theta: polar angle in the range [0,pi].
    phi: azimuthal angle in the range [-pi/2, pi/2]
    """
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
        y0 = np.arange(ystart, ystart + ysign*(Y + np.ceil(np.abs(dy)) + period), period)
        x0 = np.arange(xstart, xstart + xsign*(X + np.ceil(np.abs(dx)) + period), period)

    else: # pi/4 < theta < 3pi/4
        if np.abs(phi) <= np.pi/4 or np.abs(phi) >= 3*np.pi/4:
            # increment y and z
            dz = X * np.tan(np.pi/2 - theta)
            dx = xsign * X
            dy = dx * np.tan(phi)
            zstart = 0 if zsign == 1 else Zhat + 2*np.ceil(np.abs(dz))
            # padz = int(np.ceil(np.abs(dz)))
            ystart = 0 if zsign == 1 else Y + 2*np.ceil(np.abs(dy))
            # pady = int(np.ceil(np.abs(dy)))
            xstart = 0 if xsign == 1 else X
            # padx = 0
            z0 = np.arange(zstart, zstart + zsign*(Zhat + np.ceil(np.abs(dz)) + period), period)
            y0 = np.arange(ystart, ystart + ysign*(Y + np.ceil(np.abs(dy)) + period), period)
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
            z0 = np.arange(zstart, zstart + zsign*(Zhat + np.ceil(np.abs(dz)) + period), period)
            y0 = np.array([ystart])
            x0 = np.arange(xstart, xstart + xsign*(X + np.ceil(np.abs(dx)) + period), period)
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


def parallel_lines_3D_v01(angles, nI, period, width=1.0, noise=0.0, display=False):
    """ Create a 3D image with parallel lines at specified angles.

    Parameters
    ----------
    angles: list of tuples
        polar [0,pi] and azimuthal [-pi/2,pi/2] angles in radians
    nI: tuple
        shape of the image (z,y,x) where y=x
    period: float
        space between lines in defined units.
    width: float
        width of the lines in defined units.
    noise: float
        magnitude of the noise added to the image.
    display: bool
        if True, return the labels as well as the image.

    Returns
    -------
    I: numpy array
    labels: numpy array
    extent: tuple
    """
    dI = [nI[1]/nI[0], 1.0, 1.0]
    xI = [np.arange(n)*d for n,d in zip(nI,dI)]
    XI = np.stack(np.meshgrid(*xI, indexing='ij'), axis=-1)
    # create the image with added noise and the mask and labels
    np.random.seed(1)
    I = np.random.randn(*nI)*noise
    labels = np.zeros(nI+[2,])
    img_borders = np.array([nI[1]-dI[0],   nI[1]-1, nI[1]-1])
    inv_sigma = np.array([[1/(dI[0]*width), 0],
                  [0, 1/(dI[1]*width)]])
    norm = 2*np.pi * np.sqrt(np.sum(1/np.diag(inv_sigma))) 
    line_thresh = np.exp(-0.5)/norm
    for i in range(len(angles)):
        theta = angles[i][0]
        phi = angles[i][1]
        # draw lines defined by theta and phi and a point on the line
        v = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])[None]
        theta_ = theta if np.abs(theta) <= np.pi/2 else np.pi - np.abs(theta)
        step = np.abs(np.array([period/(np.sin(theta_)+np.finfo(float).eps),
                                period/(np.cos(theta_*np.cos(phi))+np.finfo(float).eps),
                                period/(np.cos(theta_*np.sin(phi)))+np.finfo(float).eps
                                ]))
        # draw the line through the center of the images
        c0 = np.array([(n-1)*d/2 for n,d in zip(nI,dI)]) # center of the image
        line_count = 0
        # choose a plane for the line centers
        if np.abs(theta) < np.pi/4 or np.abs(theta) > 3*np.pi/4:
            print('increment in xy plane')
            # increment x and y
            boundaries = np.abs(np.array([0., np.cos(phi)*np.tan(theta)*img_borders[0]/2, np.sin(phi)*np.tan(theta)*img_borders[0]/2]))
            centers1 = np.concatenate( (np.arange(c0[1] - step[1], -boundaries[1], -step[1]), np.arange(c0[1], boundaries[1]+img_borders[1], step[1])) )
            centers2 = np.concatenate( (np.arange(c0[2] - step[2], -boundaries[2], -step[2]), np.arange(c0[2], boundaries[2]+img_borders[2], step[2])) )
            centers = np.stack(np.meshgrid(centers1, centers2, indexing='ij'), axis=-1)
            centers = np.concatenate((np.ones(centers.shape[:-1]+(1,))*c0[0], centers), axis=-1).reshape(-1,3)

        elif np.abs(phi) < np.pi/4 or np.abs(phi) > 3*np.pi/4:
            # increment x and z
            print('increment in xz plane')
            boundaries = np.abs(np.array([img_borders[1]/2/np.cos(phi)/np.tan(theta), 0., img_borders[1]/2*np.tan(phi)]))
            centers0 = np.concatenate( (np.arange(c0[0] - step[0], -boundaries[0], -step[0]), np.arange(c0[0], boundaries[0]+img_borders[0], step[0])) )
            centers2 = np.concatenate( (np.arange(c0[2] - step[2], -boundaries[2], -step[2]), np.arange(c0[2], boundaries[2]+img_borders[2], step[2])) )
            centers = np.stack(np.meshgrid(centers0, centers2, indexing='ij'), axis=-1)
            centers = np.concatenate((centers[...,0,None], np.ones(centers.shape[:-1]+(1,))*c0[1], centers[...,1,None]), axis=-1).reshape(-1,3)

        else:
            # increment y and z
            print('increment in yz plane')
            boundaries = np.abs(np.array([img_borders[2]/2/np.sin(phi)/np.tan(theta), img_borders[2]/2/np.tan(phi), 0.]))
            centers0 = np.concatenate( (np.arange(c0[0] - step[0], -boundaries[0], -step[0]), np.arange(c0[0], boundaries[0]+img_borders[0], step[0])) )
            centers1 = np.concatenate( (np.arange(c0[1] - step[1], -boundaries[1], -step[1]), np.arange(c0[1], boundaries[1]+img_borders[1], step[1])) )
            centers = np.stack(np.meshgrid(centers0, centers1, indexing='ij'), axis=-1)
            centers = np.concatenate((centers, np.ones(centers.shape[:-1]+(1,))*c0[2]), axis=-1).reshape(-1,3)
    
        for c in centers:
            if display:
                I, labels = draw_line_3D(I, XI, direction=v, point=c, inv_sigma=inv_sigma, norm=norm, labels=labels, line_thresh=line_thresh, display=True, angles=[theta,phi])
                line_count += 1
            else:
                I, _ = draw_line_3D(I, XI , v, c, inv_sigma, norm)
                line_count += 1
            # projection = (XI - c) - ((XI - c)@v.T)*v
            # xy_dist = np.linalg.norm(projection[...,1:], axis=-1)
            # z_dist = projection[...,0]
            # dist = np.stack([z_dist,xy_dist],-1)[...,None,:]
            # line = np.exp(-0.5 * dist @ inv_sigma @ dist.transpose(0,1,2,4,3)).squeeze() / norm
            # I += line
            # if display:
            #     line_label = np.where(line[...,None] > line_thresh, angles, [0.0,0.0])
            #     labels += line_label

            # line_count += 1


    if display:
        labels = np.where(np.any(labels[...,None,:] == np.array(angles), axis=3), labels, [0.0,0.0])

    print(f'line count: {line_count}')
    return I, labels


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


def circle(nI, period=6, width=1, noise=0.05, blur=0.0, min_radius=4): #, mask_thresh=0.5):
    # create an isotropic image first and downsample later
    # get largest dimension
    dI = [nI[1]/nI[0], 1.0]
    maxdim = np.argmax(nI)
    nIiso  = [nI[maxdim]]*len(nI)
    xI = [np.arange(n) - (n-1)//2 for n in nIiso]
    XI = np.stack(np.meshgrid(*xI,indexing='ij'),axis=-1)
    theta = np.arctan(XI[:,::-1,1]/(XI[:,::-1,0]+np.finfo(float).eps))
    # theta = np.arctan2(XI[:,::-1,1], XI[:,::-1,0])
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


def anisotropy_correction(image, dI, labels=None, direction='up', interpolation=cv2.INTER_AREA, blur=False):

    # downsample all dimensions to largest dimension or upsample to the smallest dimension.
    if direction == 'down':
        dim = np.argmax(dI)
    elif direction == 'up':
        dim = np.argmin(dI)
    dsize = [image.shape[dim]]*len(image.shape)
    image = torch.tensor(image)[None,None]
    mode = 'bilinear' if image.dim() == 4 else 'trilinear'
    upsample = Upsample(size=dsize, mode=mode, align_corners=True)
    image_corrected = upsample(image).squeeze().numpy()
    # image_corrected = cv2.resize(image, dsize=dsize, interpolation=interpolation)
    # mask_corrected = cv2.resize(mask, dsize=dsize, interpolation=cv2.INTER_NEAREST)
    # mask_corrected = np.where(mask_corrected > 0.0, 1.0, 0.0)
    labels_corrected = None
    if labels is not None:
        thetas = np.unique(labels)
        # labels_corrected = cv2.resize(labels, dsize=dsize, interpolation=interpolation)
        labels_corrected = torch.tensor(labels)[None,None]
        labels_corrected = upsample(labels_corrected).squeeze().numpy()
        # set all values in labels_corrected to the nearest value in thetas
        label_ids = np.abs(labels_corrected[:, :, None] - thetas[None, None, :]).argmin(axis=2)
        labels_corrected = thetas[label_ids]
    dI = [dI[dim]]*len(image.shape)
    xI = [(np.arange(n) - (n-1)/2)*d for n,d in zip(image_corrected.shape,dI)]
    extent = (xI[1][0]-dI[1]/2, xI[1][-1]+dI[1]/2, xI[0][-1]+dI[0]/2, xI[0][0]-dI[0]/2) # TODO: generalize for 3D case
    if blur is not False:
        image_corrected = gaussian_filter(image_corrected, sigma=blur)
    
    return image_corrected, labels_corrected, extent


def periodic_mean(points, period=180):
    period_2 = period/2
    if max(points) - min(points) > period_2:
        _points = np.array([0 if x > period_2 else 1 for x in points]).reshape(-1,1)
        n_left =_points.sum()
        n_right = len(points) - n_left
        if n_left >0:
            mean_left = (points * _points).sum()/n_left
        else:
            mean_left =0
        if n_right >0:
            mean_right = (points * (1-_points)).sum() / n_right
        else:
            mean_right = 0
        _mean = (mean_left*n_left+mean_right*n_right+n_left*period)/(n_left+n_right)
        return _mean % period
    else:
        return points.mean(axis=0)
    
def gather(I, patch_size=None):
    """ Gather I into patches.

        Parameters
        ----------
        I : three or four-dimensional array with last dimension of size n_features

        patch_size : int, or {list, tuple} of length I.ndim
            The side length of each patch

        Returns
        -------
        I_patches : four or five-dimensional array with samples aggregated in the second
            to last dimension and the last dimension has size n_features.
    """
    if patch_size is None:
        patch_size = [I.shape[1] // 10] * (I.ndim-1) # default to ~100 tiles in an isostropic image
    elif isinstance(patch_size, int):
        patch_size = [patch_size] * (I.ndim-1)
    n_features = I.shape[-1]
    if I.ndim == 3:
        i, j = [x//patch_size[i] for i,x in enumerate(I.shape[:2])]
        I_patches = I[:i*patch_size[0],:j*patch_size[1]].copy() # crop so 'I' divides evenly into patch_size (must create a new array to change stride lengths)
        # reshape into patches by manipulating strides. (np.reshape preserves contiguity of elements, which we don't want in this case)
        nbits = I_patches.strides[-1]
        I_patches = np.lib.stride_tricks.as_strided(I_patches, shape=(i,j,patch_size[0],patch_size[1],n_features),
                                                    strides=(patch_size[0]*I_patches.shape[1]*n_features*nbits,
                                                             patch_size[1]*n_features*nbits,
                                                             I_patches.shape[1]*n_features*nbits,
                                                             n_features*nbits,
                                                             nbits))
        I_patches = I_patches.reshape(i,j,np.prod(patch_size),n_features)
    elif I.ndim == 4:
        i, j, k = [x//patch_size[i] for i,x in enumerate(I.shape[:3])]
        I_patches = np.array(I[:i*patch_size[0], :j*patch_size[1], :k*patch_size[2]])
        nbits = I_patches.strides[-1]
        I_patches = np.lib.stride_tricks.as_strided(I_patches, shape=(i, j, k, patch_size[0], patch_size[1], patch_size[2], n_features),
                                                strides=(patch_size[0]*I_patches.shape[1]*I_patches.shape[2]*n_features*nbits,
                                                        patch_size[1]*I_patches.shape[2]*n_features*nbits,
                                                        patch_size[2]*n_features*nbits,
                                                        I_patches.shape[1]*I_patches.shape[2]*n_features*nbits,
                                                        I_patches.shape[2]*n_features*nbits,
                                                        n_features*nbits,
                                                        nbits))
        I_patches = I_patches.reshape(i,j,k,np.prod(patch_size),n_features)
    return I_patches

def phantom_test(derivative_sigma, tensor_sigma, phantom=None, nI=(64,64), period=10, width=1.0, noise=0.001,\
                 phantom_type='grid', err_type='piecewise', grid_thetas=None, patch_size=None, crop=None, blur_correction=False, display=False, return_all=False):
    """Test structure tensor analysis on a grid of crossing lines.

    Parameters
    ----------
    derivative_sigma : list, float
        Sigma for the derivative filter.

    tensor_sigma : float
        Sigma for the structure tensor filter.
    
    phantom : two or three dimensional ndarray , optional
        Optional previously generated phantom.

    nI : tuple of int
        Number of pixels in each dimension.

    period : int
        Space between lines.

    width : int
        Width of the lines.

    noise : float
        Noise level.

    phantom_type : {'grid', 'circles'}
        Option 'grid' generates an image filled with parallel lines at specified angles.
        Option 'circles generates an image of concentric circles.

    err_type : {'pixelwise', 'piecewise'}
        Pixelwise returns average angular difference per pixel. 
        Piecewise divides the image into pieces and fits each group of pixels
        to an ODF from which peaks are computed and compared to ground truth.

    grid_thetas : tuple of float
        Angles of the lines in radians in the range [-pi/2, pi/2]. Takes one or two line orientations.
    
    patch_size : int, or {list, tuple} of length I.ndim
        The side length of each patch
    
    crop : int
        Number of pixels to crop from the edges after anisotropy correction.

    blur_correction : bool
        If True, apply a Gaussian filter to the image to create isotropic blur.
    
    display : bool

    return_all : bool
        If True, return error, mean angle values, image, labels, angles,
        and diff (array of differences between mean and ground truth per patch).
    
    Returns
    -------
    error : float
        Average angular difference between ground truth and estimated angles for
        pixelwise error, or jensen-shannon divergence for piecewise error.

    """
    if phantom is not None:
        I = phantom
        nI = I.shape
    dim = len(nI)

    if dim == 2:

        dI = (nI[1]/nI[0], 1.0)
        # generate a new phantom if necessary.
        if phantom is None:
            assert phantom_type in ('grid', 'circles')
            assert err_type in ('pixelwise', 'piecewise')
            if phantom_type == 'grid':
                assert isinstance(grid_thetas, (list, tuple)), 'grid_thetas must be a list or tuple of angles when using a grid phantom.'
                assert np.alltrue([np.abs(theta) <= np.pi/2 for theta in grid_thetas]),\
                'thetas must be in the range [-pi/2, pi/2]'
                I, labels, extent = parallel_lines_2d_v01(grid_thetas, nI, noise=noise, period=period, display=display)
                masked = False
            elif phantom_type == 'circles':
                I, labels, extent = circle(nI, period=period, width=width, noise=noise)
                masked = False
            # apply anisotropy correction
            if blur_correction:
                I, labels, extent = anisotropy_correction(I, labels, dI, blur=(0.,dI[0]-dI[1]))
            else:
                I, labels, extent = anisotropy_correction(I, labels, dI)
            dI = (1.0, 1.0)

        # compute structure tensor and angles
        S = histology.structure_tensor(I, derivative_sigma=derivative_sigma, tensor_sigma=tensor_sigma, dI=dI, masked=masked)
        angles = histology.angles(S)

        # compute error
        if err_type == 'pixelwise':

            # compute the average difference between computed angles and labels
            angles_flipped = np.where(angles < 0, angles + np.pi, angles - np.pi)
            angles_ = np.stack((angles, angles_flipped), axis=-1)
            labels_flipped = np.where(labels < 0, labels + np.pi, labels - np.pi)
            labels_ = np.stack((labels, labels_flipped), axis=-1)
            diff1 = np.abs(angles_ - labels_)
            diff2 = np.abs(angles_[...,::-1] - labels_)
            diff = np.concatenate((diff1, diff2), axis=-1)
            diff = np.nanmin(diff, axis=-1)
            error = np.nanmean(diff) * 180 / np.pi # average error in radians

        elif err_type == 'piecewise':

            # first crop boundaries to remove artifacts related to averaging tensors near the edges.
            if crop is None:
                crop = 15
            angles = angles[crop:-crop, crop:-crop]
            if patch_size is not None:
                # gather angles into non-overlapping patches
                angles_ = gather(angles, patch_size=patch_size)
                angles_ = angles_.squeeze(axis=-1)
            else:
                angles_ = angles.reshape(-1,dim)[None,None]

            # Estimate kmeans centers and errors for each tile.
            grid_thetas = np.array(grid_thetas)
            diff = np.zeros(angles_.shape[:2])
            for i in range(angles_.shape[0]):
                for j in range(angles_.shape[1]):
                    angles_tile = angles_[i,j][~np.isnan(angles_[i,j])]
                    angles_tile = np.where(angles_tile < 0, angles_tile + np.pi, angles_tile) # flip angles to be in the range [0,pi] for periodic kmeans
                    if len(grid_thetas) == 1:
                        mu_ = periodic_mean(angles_tile.flatten()[...,None], period=np.pi)
                    else:
                        periodic_kmeans = PeriodicKMeans(angles_tile[...,None], period=np.pi, no_of_clusters=2)
                        _, _, centers = periodic_kmeans.clustering()
                        mu_ = np.array(centers).squeeze()
                    mu_flipped = np.where(mu_ < 0, mu_ + np.pi, mu_ - np.pi)
                    mu = np.stack((mu_,mu_flipped), axis=-1)
                    # diff[i,j] = np.mean(np.min(np.abs(mu[...,None] - grid_thetas),axis=(-1,-2)), axis=-1) * 180/np.pi
                    diff_ = np.abs(mu[...,None] - grid_thetas) # this has shape (2,2,2) for 2 mu values each with 2 possible orientations, and each compared to both ground truth angles
                    if len(grid_thetas) == 1:
                        diff[i,j] = np.min(diff_) * 180/np.pi
                    else:
                        argmin = np.array(np.unravel_index(np.argmin(diff_), (2,2,2))) # the closest mu value and orientation is the first error
                        remaining_idx = 1 - argmin # the second error is the best error from the other mu value compared to the other ground truth angle
                        diff[i,j] = np.mean([diff_[tuple(argmin)], np.min(diff_,1)[remaining_idx[0],remaining_idx[2]]]) * 180/np.pi
            error = np.mean(diff)

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

    elif dim == 3:

        dI = (nI[1]/nI[0], 1.0, 1.0)
        if phantom is None:
            assert isinstance(grid_thetas, (list, tuple)), 'grid_thetas must be a list or tuple of angles when using a grid phantom.'
            assert len(grid_thetas[0]) == 2, 'each angle must have a polar and azimuthal component.'
            I, labels = parallel_lines_3D_v01(grid_thetas, nI, noise=noise, period=period, width=width, display=display)
            # apply anisotropy correction
            if blur_correction:
                sigma = dI[0]-dI[1]
                I, _, extent = anisotropy_correction(I, dI, blur=(0., sigma, sigma))
            else:
                I, _, extent = anisotropy_correction(I, dI)
            dI = (1.0, 1.0, 1.0)

        # compute structure tensor and angles
        S = histology.structure_tensor(I, derivative_sigma=derivative_sigma, tensor_sigma=tensor_sigma, dI=dI, masked=False)
        angles = histology.angles(S, cartesian=True) # shape is (i,j,k,3) where the last dimension is in x,y,z order
        # crop boundaries to remove artifacts related to averaging tensors near the edges.
        if crop is None:
            crop = 15
        elif crop > 0:
            angles = angles[:,crop:-crop, crop:-crop, crop:-crop]
        if patch_size is not None:
            # gather angles into non-overlapping patches
            angles_ = gather(angles, patch_size=patch_size)
        else:
            angles_ = angles.reshape(-1,dim)[None,None,None]
        # convert grid_thetas to cartesian coordinates for easier error calculation
        grid_thetas = np.array(grid_thetas)
        grid_thetas = np.array([np.sin(grid_thetas[:,0])*np.sin(grid_thetas[:,1]),
                                np.sin(grid_thetas[:,0])*np.cos(grid_thetas[:,1]),
                                np.cos(grid_thetas[:,0])
                                ]).T # shape (n_clusters, n_features)
        # Estimate kmeans centers for each tile.
        if len(grid_thetas) == 1:
            skm = apsym_kmeans.APSymKMeans(n_clusters=1)
        else:
            skm = apsym_kmeans.APSymKMeans(n_clusters=2)
        diff = np.empty(angles_.shape[:3])
        for i in range(angles_.shape[0]):
            for j in range(angles_.shape[1]):
                for k in range(angles_.shape[2]):
                    if len(grid_thetas) == 1:
                        # mu_ = np.mean(angles_[i,j,k], axis=0)
                        # mu_ = mu_ / np.linalg.norm(mu_)
                        skm.fit(angles_[i,j,k])
                        mu_ = skm.cluster_centers_
                        diff[i,j,k] = np.arccos(np.abs(mu_.dot(grid_thetas[0]))) * 180/np.pi 
                    else:
                        skm.fit(angles_[i,j,k])
                        mu_ = skm.cluster_centers_ # shape (n_clusters, n_features)
                        diff_ = np.empty((len(mu_),len(grid_thetas))) # shape (2,2) for two permutations of the difference between two means and two grid_thetas
                        for m in range(len(mu_)):
                            for n in range(len(grid_thetas)):
                                diff_[m,n] = np.arccos(np.abs(mu_[m].dot(grid_thetas[n])))
                        argmax = np.unravel_index(np.argmin(diff_), (2,2))
                        corrolary = tuple([1 - x for x in argmax]) # the corresponding cos_dif of the other mu to the other grid_theta
                        diff[i,j,k] = np.mean([diff_[argmax], diff_[corrolary]]) * 180/np.pi

        error = np.mean(diff)

        if display:
            labels = np.array(labels)
            labels = np.array([np.sin(labels[...,0])*np.sin(labels[...,1]),
                                    np.sin(labels[...,0])*np.cos(labels[...,1]),
                                    np.cos(labels[...,0])
                                    ]).T # shape (n_clusters, n_features)
            if angles_.shape[0] > 1:
                fig, ax = plt.subplots(4,3, figsize=(4,3))
            else:
                fig, ax = plt.subplots(3,3, figsize=(3,3))
            ax[0][0].imshow(I[nI[1]//2], extent=extent)
            ax[0][0].set_title('Image xy')
            ax[0][1].imshow(I[:,nI[1]//2], extent=extent)
            ax[0][1].set_title('Image zx')
            ax[0][2].imshow(I[:,:,nI[1]//2], extent=extent)
            ax[0][2].set_title('Image zy')
            ax[1][0].imshow(np.abs(labels[nI[1]//2]), extent=extent)
            ax[1][0].set_title('labels xy')
            ax[1][1].imshow(np.abs(labels[:,nI[1]//2]), extent=extent)
            ax[1][1].set_title('labels zx')
            ax[1][2].imshow(np.abs(labels[:,:,nI[1]//2]), extent=extent)
            ax[1][2].set_title('labels zy')
            ax[2][0].imshow(np.abs(angles[nI[1]//2]), extent=extent)
            ax[2][0].set_title('angles xy')
            ax[2][1].imshow(np.abs(angles[:,nI[1]//2]), extent=extent)
            ax[2][1].set_title('angles zx')
            ax[2][2].imshow(np.abs(angles[:,:,nI[1]//2]), extent=extent)
            ax[2][2].set_title('angles zy')
            if angles_.shape[0] > 1:
                ax[3][0].imshow(diff[diff.shape[0]//2], extent=extent)
                ax[3][0].set_title('diff xy')
                ax[3][1].imshow(diff[:,diff.shape[1]//2], extent=extent)
                ax[3][1].set_title('diff zx')
                ax[3][2].imshow(diff[:,:,diff.shape[1]//2], extent=extent)
                ax[3][2].set_title('diff zy')
            else:
                print(f'error = {diff[0,0,0]} degrees')
            plt.show()

    if return_all:
        return error, mu_, I, labels, angles, diff
    else:
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

##########################################################################################################################################
# phantom_tests methods for computing error on grid phantoms

# # compare odf to the ground truth distribution using jenson-shannon divergence.
# # the ground truth is a distribution of delta functions at the angles of the lines.
# thetas_symmetric = np.concatenate((grid_thetas, np.array(grid_thetas) + np.pi))
# thetas_symmetric = np.where(thetas_symmetric > np.pi, thetas_symmetric - 2*np.pi, thetas_symmetric)
# ground_truth = np.zeros(odf.shape[-1])
# ground_truth[np.digitize(thetas_symmetric, sample_points)] = 1.0 / len(thetas_symmetric)
# # ground_truth = gaussian_filter(ground_truth, sigma=1)
# js = np.apply_along_axis(lambda a: scipy.spatial.distance.jensenshannon(a, ground_truth), axis=-1, arr=odf)
# error = np.mean(js)

# # The below method produces errors because scipy.signal.find_peaks does not always find the exact number of peaks as thetas.
# odf_peaks = np.apply_along_axis(lambda a: scipy.signal.find_peaks(a, prominence=0.005)[0], axis=-1, arr=odf)
# # take only angles in the range -pi/2, pi/2
# peaks = np.apply_along_axis(lambda a: [p for p in sample_points[a] if p >= -np.pi/2 and p <= np.pi/2 ], axis=-1, arr=odf_peaks)
# assert peaks.shape[-1] == len(thetas), 'number of peaks detected does not equal the number of angles in the image'
# thetas.sort() # peaks are already be in order from least to greatest
# error.append( np.sum(np.abs(peaks - thetas)) / peaks.shape )

##########################################################################################################################################
# previous 3D line drawing algorithm

# step_0 = np.abs(np.array([0., period / np.cos(theta*np.cos(phi)), 0.]))
# step_1 = np.abs(np.array([0., 0., period / np.cos(theta*np.sin(phi))]))
# shift_fast = np.array([0.,0.,0.])
# shift_slow = np.array([0.,0.,0.])
# c_array = [c0]
# while 1:
#     # shift_fast += step_perpendicular
#     shift_fast += step_0
#     out_of_bounds = np.any(c0+shift_fast+shift_slow > boundaries+img_borders) #or (np.abs(c0+shift_fast+shift_slow)[1] > img_borders[1] and np.abs(c0+shift_fast+shift_slow)[2] > img_borders[2])
#     if not out_of_bounds:
#         # draw line
#         if display:
#             I, labels = draw_line_3D(I, XI, direction=v, point=c0+shift_fast+shift_slow, inv_sigma=inv_sigma, norm=norm, labels=labels, line_thresh=line_thresh, display=True, angles=[theta,phi])
#             I, labels = draw_line_3D(I, XI, direction=v, point=c0-shift_fast+shift_slow, inv_sigma=inv_sigma, norm=norm, labels=labels, line_thresh=line_thresh, display=True, angles=[theta,phi])
#             I, labels = draw_line_3D(I, XI, direction=v, point=c0+shift_fast-shift_slow, inv_sigma=inv_sigma, norm=norm, labels=labels, line_thresh=line_thresh, display=True, angles=[theta,phi])
#             I, labels = draw_line_3D(I, XI, direction=v, point=c0-shift_fast-shift_slow, inv_sigma=inv_sigma, norm=norm, labels=labels, line_thresh=line_thresh, display=True, angles=[theta,phi])
#             line_count += 4

#         else:
#             I, _ = draw_line_3D(I, XI , v, c0+shift_fast+shift_slow, inv_sigma, norm)
#             I, _ = draw_line_3D(I, XI , v, c0-shift_fast+shift_slow, inv_sigma, norm)
#             I, _ = draw_line_3D(I, XI , v, c0+shift_fast-shift_slow, inv_sigma, norm)
#             I, _ = draw_line_3D(I, XI , v, c0-shift_fast-shift_slow, inv_sigma, norm)
#             c_array += [c0+shift_fast+shift_slow, c0-shift_fast+shift_slow, c0+shift_fast-shift_slow, c0-shift_fast-shift_slow]
#             line_count += 4

#     else:
#         # shift_slow += step_parallel
#         shift_slow += step_1
#         shift_fast = np.array([0.,0.,0.])
#         out_of_bounds = np.any(c0+shift_slow > boundaries+img_borders) #or (np.abs(c0+shift_fast+shift_slow)[1] > img_borders[1] and np.abs(c0+shift_fast+shift_slow)[2] > img_borders[2])
#         if not out_of_bounds:
#             # draw line
#             if display:
#                 I, labels = draw_line_3D(I, XI, direction=v, point=c0+shift_slow, inv_sigma=inv_sigma, norm=norm, labels=labels, line_thresh=line_thresh, display=True, angles=[theta,phi])
#                 I, labels = draw_line_3D(I, XI, direction=v, point=c0-shift_slow, inv_sigma=inv_sigma, norm=norm, labels=labels, line_thresh=line_thresh, display=True, angles=[theta,phi])
#                 line_count += 2
#             else:
#                 I, _ = draw_line_3D(I, XI , v, c0+shift_slow, inv_sigma, norm)
#                 I, _ = draw_line_3D(I, XI , v, c0-shift_slow, inv_sigma, norm)
#                 c_array += [c0+shift_slow, c0-shift_slow]
#                 line_count += 2
#         else:
#             c_array = np.array(c_array)
#             break