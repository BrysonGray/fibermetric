'''
Author: Bryson Gray
2022

'''

# %%
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter, sobel
from torch.nn.functional import grid_sample
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from skimage.transform import resize
import numpy as np
import torch
import matplotlib
import argparse
import h5py

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

# function for applying rigid transformations to S
def construct_S(yy, xx, xy, down=0, A=None):
    '''
    Construct the structure tensor image from its components and apply a rigid transfrom if one is given.
    
    Parameters
    ----------
    xx : numpy 2d-array
        Numpy array storing x gradient information of the original 2D image.
    yy : numpy 2d-array
        Numpy array storing y gradient information of the original 2D image.
    xy : numpy 2d-array
        Numpy array storing x gradient * y gradient of the original 2D image.
    A : numpy 2d-array
        3x3 rigid transformation matrix
    Returns
    -------
    S : array
        Structure tensor image
    
    
    '''
    if A is not None:
        # first apply rigid transformation to coordinates and resample each component of S.
        Ai = np.linalg.inv(A)
        # get coordinate grid of S
        xS = [np.arange(xx.shape[0])-xx.shape[0]/2, np.arange(xx.shape[1])-xx.shape[1]/2, np.arange(2)] # place origin in center of image
        XS = np.stack(np.meshgrid(xS[0], xS[1], xS[2]), axis=-1).transpose(1,0,2,3) # we assume A is in x,y,z order so make XS x,y,z to match
        # apply transform to coordinate grid
        AiS = (Ai @ XS[..., None])[...,0].transpose(3,0,1,2)
        # resample each component on transformed grid
        xx = interp(xS, xx[None, ..., None].astype(np.double), AiS).squeeze().numpy()[...,0]
        yy = interp(xS, yy[None, ..., None].astype(np.double), AiS).squeeze().numpy()[...,0]
        xy = interp(xS, xy[None, ..., None].astype(np.double), AiS).squeeze().numpy()[...,0]

    S = np.stack((yy, xy, xy, xx), axis=-1)
    if down:
        S = resize(S, (S.shape[0]//down, S.shape[1]//down, 4), anti_aliasing=True)
    S = S.reshape((S.shape[0],S.shape[1],2,2))
    
    # transform structure tensors
    if A is not None:
        S = A[:2,:2] @ S @ np.transpose(A[:2,:2])  # transform structure tensors
    
    return S

def struct_tensor(I, sigma=3, down=0, A=None, all=False):
    print('calculating image gradients...')
    # I_x = gaussian_filter1d(I, sigma=sigma, axis=1, order=1)
    # I_x = gaussian_filter1d(I_x, sigma=sigma, axis=0, order=0)
    # I_y = gaussian_filter1d(I, sigma=sigma, axis=0, order=1)
    # I_y = gaussian_filter1d(I_y, sigma=sigma, axis=1, order=0)

    # # construct the structure tensor, s
    # print('constructing structure tensors...')
    # S = construct_S(I_y**2, I_x**2,
    #     I_x*I_y, A=A, down=down)
    
    I_y = sobel(I, axis=0)
    I_x = sobel(I, axis=1)

    # construct the structure tensor, s
    print('constructing structure tensors...')
    I_y_sq = gaussian_filter(I_y**2, sigma=sigma)
    I_x_sq = gaussian_filter(I_x**2, sigma=sigma)
    I_xy = gaussian_filter(I_x*I_y, sigma=sigma)

    S = construct_S(I_y_sq, I_x_sq,
        I_xy, A=A, down=down)
    if all:
        # construct orientation (theta) and anisotropy index (AI)
        print('calculating orientations and anisotropy...')
        d,v = np.linalg.eigh(S)

        # get only principle eigenvectors
        max_idx = np.abs(d).argmax(axis=-1)
        max_idx = np.ravel(max_idx)
        max_idx = np.array([np.arange(max_idx.shape[0]), max_idx])
        v = np.transpose(v, axes=(0,1,3,2)).reshape(-1,2,2)
        v = v[max_idx[0],max_idx[1]].reshape(S.shape[0],-1,2)
        theta = ((np.arctan(v[...,1] / v[...,0])) + np.pi / 2) / np.pi
        AI = abs(d[...,0] - d[...,1]) / abs(d[...,0] + d[...,1])

        # make hsv image where hue= primary orientation, saturation= anisotropy, value= original image
        print('constructing hsv image...')
        
        if down:
            I = resize(I, (I.shape[0]//down, I.shape[1]//down), anti_aliasing=True)
        stack = np.stack([theta,AI,I], -1)
        hsv = matplotlib.colors.hsv_to_rgb(stack)

        return {'S':S, 'theta':theta, 'AI':AI, 'hsv':hsv}
    else:
        return S

def odf2d(theta, nbins, tile_size, damping=0.1):
    '''
    Create an array of 2D orientation distribution functions by aggregating the structure tensor angles (theta) into tiles.
    The odf is modeled as a fourier series fit to the histogram of angles for each tile. The length of the output odf is set to nbins. 

    Parameters
    ----------
    theta: numpy Array
        Two dimensional array of angles scaled to the range (0,1).
    nbins: int
        The length of each odf in the output array.
    tile_size: int
        The size of the tiles used to aggregate angles. The sample size for each odf will be tile_size^2.
    damping: int
        Sets the degree of damping applied to fourier coefficients. Greater damping results in smoother odfs.

    '''
    # reshape theta so the last two dimensions are shape (patch_size, patch_size)
    # it will have to be cropped if the number of pixels on each dimension doesn't divide evenly into patch_size
    i, j = [x//tile_size for x in theta.shape]
    theta = theta[:i*tile_size,:j*tile_size].reshape((i,j,tile_size*tile_size))
    theta_flip = np.where(theta <= 0, theta+1, theta-1)
    theta = np.concatenate((theta,theta_flip), axis=-1)
    theta_F = np.fft.rfft(theta)
    n = np.arange(len(theta_F))
    theta_F = theta_F * np.exp(-1*damping*n) # apply damping
    odf = np.fft.irfft(theta_F, nbins)
    odf = odf / np.sum(odf, axis=-1)[...,None] # normalize to make this a pdf

    return odf

def interp(x,I,phii,**kwargs):
    '''
    Interpolate a 3D image with specified regular voxel locations at specified sample points.
    
    Interpolate the 3D image I, with regular grid positions stored in x (1d arrays),
    at the positions stored in phii (3D arrays with first channel storing component)
    
    Parameters
    ----------
    x : list of numpy arrays
        x[i] is a numpy array storing the pixel locations of imaging data along the i-th axis.
        Note that this MUST be regularly spaced, only the first and last values are queried.
    I : array
        Numpy array or torch tensor storing 3D imaging data.  I is a 4D array with 
        channels along the first axis and spatial dimensions along the last 3 
    phii : array
        Numpy array or torch tensor storing positions of the sample points. phii is a 4D array
        with components along the first axis (e.g. x0,x1,x1) and spatial dimensions 
        along the last 3.
    kwargs : dict
        Keyword arguments to be passed to the grid sample function. For example
        to specify interpolation type like nearest.  See pytorch grid_sample documentation.
    
    Returns
    -------
    out : torch tensor
        4D array storing a 3D image with channels stored along the first axis. 
        This is the input image resampled at the points stored in phii.
    
    
    '''
    # first we have to normalize phii to the range -1,1    
    I = torch.as_tensor(I)
    phii = torch.as_tensor(phii)
    phii = torch.clone(phii)
    for i in range(3):
        phii[i] -= x[i][0]
        phii[i] /= x[i][-1] - x[i][0]
    # note the above maps to 0,1
    phii *= 2.0
    # to 0 2
    phii -= 1.0
    # done
        
    # NOTE I should check that I can reproduce identity
    # note that phii must now store x,y,z along last axis
    # is this the right order?
    # I need to put batch (none) along first axis
    # what order do the other 3 need to be in?    
    out = grid_sample(I[None],phii.flip(0).permute((1,2,3,0))[None],align_corners=True,**kwargs)
    # note align corners true means square voxels with points at their centers
    # post processing, get rid of batch dimension
    out = out[0]
    return out


def read_matrix_data(fname):
    '''
    Read rigid transforms as matrix text file.
    
    Parameters
    ----------
    fname : str
    
    Returns
    -------
    A : array
        matrix in xyz order
    '''
    A = np.zeros((3,3))
    with open(fname,'rt') as f:
        i = 0
        for line in f:            
            if ',' in line:
                # we expect this to be a csv
                for j,num in enumerate(line.split(',')):
                    A[i,j] = float(num)
            else:
                # it may be separated by spaces
                for j,num in enumerate(line.split(' ')):
                    A[i,j] = float(num)
            i += 1
    
    return A

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
    
    if args.affine:
        with open(args.affine, 'rt') as f:
            A = read_matrix_data(f)

    img_down = args.img_down
    sigma = args.sigma
    down = args.down
    reverse_intensity = args.reverse_intensity

    I = load_img(image, img_down, reverse_intensity)
    if args.all==True:
        st_out = struct_tensor(I, sigma, down, all=True)
        S = st_out['S']
        theta = st_out['theta']
        AI = st_out['AI']
        hsv = st_out['hsv']
    else:
        S = struct_tensor(I, sigma, down)

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
#%%
if __name__ == "__main__":
    main()

#%%
