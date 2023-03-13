#!/usr/bin/env python

'''
Histology fiber orienataion analysis tools

Author: Bryson Gray
2022

'''

from scipy.ndimage import gaussian_filter, sobel, correlate1d
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from skimage.transform import resize
import numpy as np
import matplotlib
import argparse
import h5py
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
from dipy.reconst.shm import sh_to_sf_matrix
from fibermetric.utils import interp, read_matrix_data



def load_img(impath, img_down=0, reverse_intensity=False):
    imname = os.path.split(impath)[1]
    print(f'loading image {imname}...')
    I = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
    if img_down:
        print('downsampling image...')
        I = resize(I, (I.shape[0]//img_down, I.shape[1]//img_down), anti_aliasing=True)
    # fit I to range (0,1)
    if np.max(I[0]) > 1:
        I = I * 1/255
    if reverse_intensity == True:
        I = 1 - I

    return I


def structure_tensor(I, derivative_sigma=1.0, tensor_sigma=1.0, dI=1):
    '''
    Construct structure tensors from a grayscale image. Accepts 2D or 3D arrays

    Parameters
    ----------
    I : array
        2D or 3D scalar image
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in which case it is equal
        for all axes.
    dI : int or tuple, optional
        Image pixel dimensions. The derivative and gaussian kernel standard deviation is scaled by the inverse of the pixel size.
        If int, all dimensions are treated as the same size.
    Returns
    -------
    S : array
        Array of structure tensors with image dimensions along the first axes and tensors in the last two dimensions.

    '''
    if I.ndim == 2:
        if type(dI) == float:
            dy, dx = dI, dI
        else:
            dy,dx = dI

        Ix =  gaussian_filter(I, sigma=[derivative_sigma/dy, derivative_sigma/dx], order=(0,1))
        # Ix = correlate1d(I, np.array([-1,0,1])/2.0/dx, 1)
        # Ix = gaussian_filter(Ix, sigma=[derivative_sigma/dy, derivative_sigma/dx])
        Iy =  gaussian_filter(I, sigma=[derivative_sigma/dy, derivative_sigma/dx], order=(1,0))
        # Iy = correlate1d(I, np.array([-1,0,1])/2.0/dy, 0)
        # Iy = gaussian_filter(Iy, sigma=[derivative_sigma/dy, derivative_sigma/dx])
        norm = np.sqrt(Ix**2 + Iy**2)
        Ix = Ix / norm
        Iy = Iy / norm

        # construct the structure tensor, s
        Ixx = gaussian_filter(Ix*Ix, sigma=[tensor_sigma/dy, tensor_sigma/dx])
        Ixy = gaussian_filter(Ix*Iy, sigma=[tensor_sigma/dy, tensor_sigma/dx])
        Iyy = gaussian_filter(Iy*Iy, sigma=[tensor_sigma/dy, tensor_sigma/dx])

        # S = np.stack((Iyy, Ixy, Ixy, Ixx), axis=-1)
        S = np.stack((1-Ixx,-Ixy,-Ixy,1-Iyy), axis=-1) # identity minus the structure tensor
        S = S.reshape((S.shape[:-1]+(2,2)))

    elif I.ndim == 3:
        if type(dI) == float:
            dz, dy, dx = dI, dI, dI
        else:
            dz, dy, dx = dI

        Ix =  gaussian_filter(I, sigma=[derivative_sigma/dz, derivative_sigma/dy, derivative_sigma/dx], order=(0,0,1))
        # Ix = correlate1d(I, np.array([-1,0,1])/2.0/dx, 2)
        # Ix = gaussian_filter(Ix, sigma=[derivative_sigma/dz, derivative_sigma/dy, derivative_sigma/dx])
        Iy =  gaussian_filter(I, sigma=[derivative_sigma/dz, derivative_sigma/dy, derivative_sigma/dx], order=(0,1,0))
        # Iy = correlate1d(I,np.array([-1,0,1])/2.0/dy,1)
        # Iy = gaussian_filter(Iy, sigma=[derivative_sigma/dz, derivative_sigma/dy, derivative_sigma/dx])
        Iz =  gaussian_filter(I, sigma=[derivative_sigma/dz, derivative_sigma/dy, derivative_sigma/dx], order=(1,0,0))
        # Iz = correlate1d(I, np.array([-1,0,1])/2.0/dz, 0)
        # Iz = gaussian_filter(Iz, sigma=[derivative_sigma/dz, derivative_sigma/dy, derivative_sigma/dx])

        norm = np.sqrt(Ix**2 + Iy**2 + Iz**2)
        Ix = Ix / norm
        Iy = Iy / norm
        Iz = Iz / norm

        Ixx = gaussian_filter(Ix*Ix, sigma=[tensor_sigma/dz, tensor_sigma/dy, tensor_sigma/dx])
        Iyy = gaussian_filter(Iy*Iy, sigma=[tensor_sigma/dz, tensor_sigma/dy, tensor_sigma/dx])
        Izz = gaussian_filter(Iz*Iz, sigma=[tensor_sigma/dz, tensor_sigma/dy, tensor_sigma/dx])
        Ixy = gaussian_filter(Ix*Iy, sigma=[tensor_sigma/dz, tensor_sigma/dy, tensor_sigma/dx])
        Ixz = gaussian_filter(Ix*Iz, sigma=[tensor_sigma/dz, tensor_sigma/dy, tensor_sigma/dx])
        Iyz = gaussian_filter(Iy*Iz, sigma=[tensor_sigma/dz, tensor_sigma/dy, tensor_sigma/dx])

        # S = np.stack((Izz, Iyz, Ixz, Iyz, Iyy, Ixy, Ixz, Ixy, Ixx), axis=-1)
        S = np.stack((1-Ixx, -Ixy, -Ixz, -Ixy, 1-Iyy, -Iyz, -Ixz, -Iyz, 1-Izz), axis=-1)
        S = S.reshape((S.shape[:-1]+(3,3)))
    else:
        raise Exception(f'Input must be a 2 or 3 dimensional array but found: {I.ndim}')

    return S

def anisotropy(w):
    """
    Calculate anisotropy from eigenvalues. Accepts 2 or 3 eigenvalues

    Parameters
    ----------
    w : array
        Array with eigenvalues along the last dimension.
    
    Returns
    --------
    A : array
        Array of anisotropy values.
    """

    if w.shape[-1] == 3:
        w = w.transpose(3,0,1,2)
        trace = np.sum(w, axis=0)
        A = np.sqrt((3/2) * (np.sum((w - (1/3)*trace)**2,axis=0) / np.sum(w**2, axis=0)))
        A = np.nan_to_num(A)
        A = A/np.max(A)
    elif w.shape[-1] == 2:
        A = abs(w[...,0] - w[...,1]) / abs(w[...,0] + w[...,1])
    else:
        raise Exception(f'Accepts 2 or 3 eigenvalues but found {w.shape[-1]}')
    
    return A


def angles(S):
    """
    Compute angles from structure tensors.

    Parameters
    ----------
    S : ndarray
        Structure tensor valued image array.
    
    Returns
    -------
    angles : ndarray
        Array of values between -pi/2 and pi/2.
    """
    
    w,v = np.linalg.eigh(S)
    v = v[...,-1] # the principal eigenvector is always the last one since they are ordered by least to greatest eigenvalue with all being > 0
    if w.shape[-1] == 2:
        theta = (np.arctan(v[...,1] / (v[...,0] + np.finfo(float).eps))) # row/col gives the counterclockwise angle from left/right direction.
        return (theta,)
    else:
        x = v[...,0]
        y = v[...,1]
        z = v[...,2]
        theta = np.arctan(-z / (np.sqrt(x**2 + y**2) + np.finfo(float).eps)) + np.pi / 2  # range is (0,pi)
        phi = np.arctan(y / (x + np.finfo(float).eps)) # range (-pi/2, pi/2)
        return (theta,phi)


def hsv(S, I):
    """
    Compute angles, anisotropy index, and hsv image from 2x2 structure tensors.

    Parameters
    ----------
    S : array
        Array of structure tensors with shape MxNx2x2
    I : array
        Image with shape MxN

    Returns
    -------
    theta : array
        Array of angles (counterclockwise from left/right) with shape MxN. Angles were mapped from [-pi/2,pi/2] -> [0,1] for easier visualization.
    AI : array
        Array of anisotropy index with shape MxN
    hsv : array
        Image with theta -> hue, AI -> saturation, and I -> value (brightness).
    """
    # check if I is 2D
    if I.ndim != 2:
        raise Exception(f'Only accepts two dimensional images but found {I.ndim} dimensions')

    print('calculating orientations and anisotropy...')
    w,v = np.linalg.eigh(S)

    # get only principle eigenvectors
    # max_idx = np.abs(d).argmax(axis=-1)
    # max_idx = np.ravel(max_idx)
    # max_idx = np.array([np.arange(max_idx.shape[0]), max_idx])
    # v = np.moveaxis(v, -1,-2).reshape(-1,2,2)
    # v = v[max_idx[0],max_idx[1]].reshape(S.shape[0],-1,2)
    v = v[...,-1] # the principal eigenvector is always the last one since they are ordered by least to greatest eigenvalue with all being > 0
    # theta = ((np.arctan(v[...,0] / v[...,1])) + np.pi / 2) / np.pi
    theta = ((np.arctan(v[...,1] / v[...,0])) + np.pi / 2) / np.pi # TODO: verify this is correct since changing S component order. 
    # row/col gives the counterclockwise angle from left/right direction. Rescaled [-pi/2,pi/2] -> [0,1]
    AI = anisotropy(w) # anisotropy index (AI)

    # make hsv image where hue= primary orientation, saturation= anisotropy, value= original image
    print('constructing hsv image...')
    
    if S.shape[:-2] != I.shape:
        down = [x//y for x,y in zip(I.shape, S.shape[:-2])]
        I = resize(I, (I.shape[0]//down[0], I.shape[1]//down[1]), anti_aliasing=True)
    stack = np.stack([theta,AI,I], -1)
    hsv = matplotlib.colors.hsv_to_rgb(stack)

    return theta, AI, hsv


def transform(S, xS, A, indexing='ij', direction='f'):
    """
    Transform structure tensors.

    Parameters
    ----------
    S : (...,2,2) array
        Structure tensor array.
    xS : list of 1D arrays
        list of points along each axis.
    A : array
        Affine transfrom matrix. May be either one matrix (2,2) or stack of affine matrices (M,2,2).
    indexing : {'ij', 'xy'}, optional
        Indexing of transform matrix. Default is 'ij'.
    direction : {'f', 'b'}, optional
        Direction of transform. If 'b', the inverse of A is applied.

    Returns
    -------
    Sr : (...,2,2) array
        Registered structure tensor array.

    """

    if S.ndim == 4:
        xS.insert(0,np.array([0]))
        S = S[None]
        A = A[None]
    S = np.stack((S[...,0,0], S[...,0,1], S[...,1,1]))
    XS = np.stack(np.meshgrid(xS[0],xS[1], xS[2], indexing='ij'), axis=-1)
    if indexing == 'xy':
        A = A[:,[1,0,2]] # first flip the first two rows
        A = A[:, :, [1,0,2]] # then the first two columns
    if direction == 'b':
        A = np.linalg.inv(A)
    points = (A[:, None, None, :2, :2] @ XS[:A.shape[0], ..., 1:, None])[..., 0] + A[:, None, None, :2, -1]
    points = points.transpose(-1,0,1,2)
    points = np.concatenate((np.ones_like(points[0])[None]*xS[0][None,:,None,None], points))
    Sr = interp(xS, S, points.astype(float))
    Sr = np.stack((Sr[0],Sr[1],Sr[1],Sr[2]),axis=-1)[0].reshape(Sr.shape[2:]+(2,2))

    return Sr


def downsample(S,down):
    """
    Downsample structure tensors.

    Parameters
    ----------
    S : array
        Array of structure tensors. Last two dimensons contain 2x2 or 3x3 tensors
    down : int or list or tuple
        Downsampling factor.
    Returns
    -------
    S : array
        Downsampled 
    """

    S = S.reshape((S.shape[:-2] + (S.shape[-1] * S.shape[-2],)))
    if S.ndim == 3:
        if type(down) == int:
            d0,d1 = down, down
        else:
            d0,d1 = down
        S = resize(S, (S.shape[0]//d0, S.shape[1]//d1, 4), anti_aliasing=True).reshape(S.shape[:-1]+(2,2))
    elif S.ndim == 4:
        if type(down) == int:
            d0,d1,d2 = down,down,down
        else:
            d0,d1,d2 = down
        S = resize(S, (S.shape[0]//d0, S.shape[1]//d1, S.shape[2]//d2, 9), anti_aliasing=True).reshape(S.shape[:-1]+(3,3))
    return S

    
def odf2d(theta, nbins, tile_size, damping=0.1):
    '''
    Create an array of 2D orientation distribution functions (ODFs) by aggregating the structure tensor angles (theta) into tiles.
    The odf is modeled as a fourier series fit to the histogram of angles for each tile. The length of the output odf is set to nbins. 

    Parameters
    ----------
    theta: numpy Array
        Two dimensional array of angles. ( range [-pi/2,pi/2] )
    nbins: int
        The length of each odf in the output array.
    tile_size: int
        The size of the tiles used to aggregate angles. The sample size for each odf will be tile_size^2.
    damping: int
        Sets the degree of damping applied to fourier coefficients. Greater damping results in smoother odfs.
    
    Returns
    -------
    odf: numpy Array
        array of ODFs with each ODF representing the distribution of angles in a patch of theta with size equal to tile_size^2.

    '''
    print('reshaping...')
    # reshape theta so the last two dimensions are shape (patch_size, patch_size)
    # it will have to be cropped if the number of pixels on each dimension doesn't divide evenly into patch_size
    i, j = [x//tile_size for x in theta.shape]
    theta = np.array(theta[:i*tile_size,:j*tile_size]) # crop so theta divides evenly into tile_size (must create a new array to change stride lengths too.)
    # reshape into tiles by manipulating strides. (np.reshape preserves contiguity of elements, which we don't want in this case)
    nbits = theta.strides[-1]
    theta = np.lib.stride_tricks.as_strided(theta, shape=(i,j,tile_size,tile_size), strides=(tile_size*theta.shape[1]*nbits,tile_size*nbits,theta.shape[1]*nbits,nbits))
    theta = theta.reshape(i,j,tile_size**2)

    print('creating symmetric distribution...')
    # concatenate a copy of theta but with angles in the opposite direction to make the odfs symmetric (since structure tensors are symmetric).
    theta_flip = theta - np.pi
    theta_flip = np.where(theta_flip < -1*np.pi, theta_flip + 2*np.pi, theta_flip)
    theta = np.concatenate((theta,theta_flip), axis=-1)

    print('fitting to fourier series...')
    t = np.arange(nbins)*2*np.pi/(nbins-1) - np.pi
    odf = np.apply_along_axis(lambda a: np.histogram(a, bins=t)[0], axis=-1, arr=theta)
    odf_F = np.fft.rfft(odf)
    n = np.arange(odf_F.shape[-1])
    odf_F = odf_F * np.exp(-1*damping*n) # apply damping
    odf = np.fft.irfft(odf_F, nbins)
    odf = odf / np.sum(odf, axis=-1)[...,None] # normalize to make this a pdf
    print('done')

    return odf


def odf3d(angles, nbins, tile_size, sh_order=8):
    '''
        Create an array of 3D orientation distribution functions (ODFs) by aggregating the structure tensor angles into tiles.
        The odf is modeled as a spherical harmonic series fit to the histogram of angles for each tile. The length of the output odf is set to nbins. 

        Parameters
        ----------
        angles: ndarray
            (i,j,k,2) array with angles theta (angle w.r.t. polar axis) and phi (azimuthal angle) in the last dimension.
        nbins: int
            Number of bins used to create histogram of angles.
        tile_size: int
            The size of the tiles used to aggregate angles. The sample size for each odf will be tile_size^3.
        sh_order: int
            Sets the largest spherical harmonic order used to fit to the angles.
        
        Returns
        -------
        sh_signal: ndarray
            Spherical harmonic signal valued image array. Contains spherical harmonic coefficients in the last dimension.

        '''

    # aggregate angles
    i, j, k = [x//tile_size for x in angles.shape[:-1]]
    angles = np.array(angles[:i*tile_size, :j*tile_size, :k*tile_size]) # crop so angles divides evenly into tile_size (must create a new array to change stride lengths too.)
    # reshape into tiles by manipulating strides. (np.reshape preserves contiguity of elements, which we don't want in this case)
    nbits = angles.strides[-1]
    angles = np.lib.stride_tricks.as_strided(angles, shape=(i, j, k, tile_size, tile_size, tile_size, 2),
                                            strides=(tile_size*angles.shape[1]*angles.shape[2]*nbits,
                                                    tile_size*angles.shape[2]*nbits,
                                                    tile_size*nbits,
                                                    angles.shape[1]*angles.shape[2]*nbits,
                                                    angles.shape[2]*nbits,
                                                    nbits, nbits//2))
    angles = angles.reshape(i * j * k, tile_size**3, 2)
    # set up approx. evenly distributed histogram bin points by despersing charges on the surface of a sphere.
    theta = np.pi * np.random.rand(nbins)
    phi = 2 * np.pi * np.random.rand(nbins)
    hsph_initial = HemiSphere(theta=theta, phi=phi)

    hsph_updated, potential = disperse_charges(hsph_initial,2)#, const=0.5)
    while np.abs(potential[-2]-potential[-1]) > 0.1:
        hsph_updated, pot = disperse_charges(hsph_updated,2)#, const=0.5)
        potential = np.append(potential,pot)

    vertices = hsph_updated.vertices
    bins = np.vstack((vertices,-vertices)) # histogram bin centers
    N = angles.shape[0] * angles.shape[1] * angles.shape[2] # number of voxels
    signal = np.zeros((N, len(bins)))
    # fill in the histogram. For each vector, find the distance to all bin centers, choose the closest bin and add one to it
    for v in angles[:]:
        dist = np.arccos(np.dot(vertices,v.T))
        v_idx = np.argmax(np.abs(np.pi/2 - dist), axis=0) # closest or farthest bin from the vector i.e. closest to v or -v
        signal[:,v_idx] += 1
        signal[:,v_idx+len(vertices)] += 1

    # fit spherical harmonics to signal
    sphere = Sphere(xyz=bins)
    print(f'Building SH matrix of order {sh_order}')
    B, invB = sh_to_sf_matrix(sphere, sh_order=sh_order)
    sh_signal = np.dot(signal, invB)
    sh_signal = sh_signal.reshape(i, j, k, sh_signal[-1])

    return sh_signal


# TODO: accept directories to process stacks of images and affines
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to image to be used for structure tensor analysis")
    parser.add_argument("-o", "--out", help="output directory")
    parser.add_argument("-a", "--all", action="store_true",
                        help="save out theta, anisotropy index, and hsv with structure tensor")
    parser.add_argument("-i", "--img_down", type=int, default=0,
                        help="image downsampling factor applied before structure tensor calculation")
    parser.add_argument("-s", "--sigma", type=float, default=1,
                        help="sigma used for gaussian filter on structure tensor")
    parser.add_argument("-d", "--down", type=int, default=1,
                        help="structure tensor downsampling factor")
    parser.add_argument("-p","--pixel_size", nargs="+", type=float, default=1.0, help="pixel dimensions")
    parser.add_argument("-r", "--reverse_intensity", action="store_true",
                        help="reverse image intensity, e.g. light on dark to dark on light")
    parser.add_argument('-A', '--affine', help=".txt file storing rigid 3x3 transformation matrix")
    args = parser.parse_args()
    
    image = args.image

    if args.out:
        out = args.out
        if not os.path.exists(out):
            os.makedirs(out)
    else:
        out = os.getcwd()
    

    img_down = args.img_down
    sigma = args.sigma
    down = args.down
    reverse_intensity = args.reverse_intensity

    I = load_img(image, img_down, reverse_intensity)
    dI = args.pixel_size

    S = structure_tensor(I, sigma, dI=dI)
    S = downsample(S, down)

    if args.all==True:
        theta, AI, hsv = hsv(S, I)

    if args.affine:
        with open(args.affine, 'rt') as f:
            A = read_matrix_data(f)
            nS = S.shape[:-2]
            if type(dI) == float:
                dI = np.ones(len(nS))*dI
            xS = [np.arange(n)*d - (n-1)*d/2.0 for n,d in zip(nS,dI)]
            S = transform(S, xS, A)

    # save out structure tensor field
    base = os.path.splitext(os.path.split(image)[1])[0]
    S_name = base + '_S.h5'
    with h5py.File(os.path.join(out, S_name), 'w') as f:
            f.create_dataset('S', data=S)

    # save out images
    if args.all:
        name = base + '_theta.png'
        cv2.imwrite(os.path.join(out, name), (theta*255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        name = base + '_AI.png'
        cv2.imwrite(os.path.join(out, name), (AI*255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        name = base + '_hsv.png'
        cv2.imwrite(os.path.join(out, name), (hsv*255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    return


if __name__ == "__main__":
    main()