# %%
from pickletools import uint8
import struct
from turtle import shape
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
def construct_S(xx, yy, xy, down=0, A=None):
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

    S = np.stack((xx, xy, xy, yy), axis=-1)
    if down:
        S = resize(S, (S.shape[0]//down, S.shape[1]//down, 4), anti_aliasing=True)
    S = S.reshape((S.shape[0],S.shape[1],2,2))
    
    # transform structure tensors
    if A is not None:
        S = np.transpose(np.linalg.inv(A[:2,:2])) @ S @ np.linalg.inv(A[:2,:2]) 
    
    return S

def struct_tensor(I, sigma=1, down=0, A=None):
    # sobel function gets approx. gradient of image intensity along the specified axis
    print('calculating image gradients...')

    I_y = sobel(I, axis=0)
    I_x = sobel(I, axis=1)

    # construct the structure tensor, s
    print('constructing structure tensors...')
    I_y_sq = gaussian_filter(I_y**2, sigma=sigma)
    I_x_sq = gaussian_filter(I_x**2, sigma=sigma)
    I_xy = gaussian_filter(I_x*I_y, sigma=sigma)

    S = construct_S(I_x_sq, I_y_sq,
        I_xy, A=A, down=down)

    # construct orientation (theta) and anisotropy index (AI)
    print('calculating orientations and anisotropy...')
    d,v = np.linalg.eig(S)

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

    return S, theta, AI, hsv


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
        keword arguments to be passed to the grid sample function. For example
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
    
    # TODO: producing the hsv image requires the original image whether S was computed previously or not. Need to figure out how to deal with this.
    image = args.image
    # if the image is a directory, assume S has been computed previously and output theta, AI and hsv.
    isdir = os.path.splitext(image)[-1] == ''

    if args.out:
        out = args.out
        if not os.path.exists(out):
            os.makedirs(out)
    else:
        out = os.getcwd()
    
    # if it is not a directory, first save out xx, yy, and xy
    if not isdir:
        if args.affine:
            with open(args.affine, 'rt') as f:
                A = read_matrix_data(f)
        img_down = args.img_down
        sigma = args.sigma
        down = args.down
        reverse_intensity = args.reverse_intensity

        I = load_img(image, img_down, reverse_intensity)
        S, theta, AI, hsv = struct_tensor(I, sigma, down)

        # save out image(s)
        base = os.path.splitext(os.path.split(image)[1])[0]
        xx_name = base + '_xx.h5'
        yy_name = base + '_yy.h5'
        xy_name = base + '_xy.h5'

        with h5py.File(os.path.join(out, xx_name), 'w') as f:
            f.create_dataset('xx', data=S[..., 0])
        with h5py.File(os.path.join(out, yy_name), 'w') as f:
            f.create_dataset('yy', data=S[..., 3])
        with h5py.File(os.path.join(out, xy_name), 'w') as f:
            f.create_dataset('xy', data=S[..., 1])

    if isdir:

        S = construct_S(xx, yy, xy, A)
        # get the base file name from the existing files
        base = os.listdir(image)[0][:-6]
    if args.all or isdir:
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

# #%%

# I_file = '/home/brysongray/data/m1229/m1229_other_data/M1229-M94--_1_0160.jp2'

# I = load_img(I_file, reverse_intensity=True)

# S1, theta, AI = struct_tensor(I, down=32)

# # #%%
# # xx_file = '/home/brysongray/structure_tensor_analysis/test_3-15/M1229-M94--_1_0160_xx.jp2'
# # yy_file = '/home/brysongray/structure_tensor_analysis/test_3-15/M1229-M94--_1_0160_yy.jp2'
# # xy_file = '/home/brysongray/structure_tensor_analysis/test_3-15/M1229-M94--_1_0160_xy.jp2'
# # xx = cv2.imread(xx_file, cv2.IMREAD_GRAYSCALE)/255
# # yy = cv2.imread(yy_file, cv2.IMREAD_GRAYSCALE)/255
# # xy = cv2.imread(xy_file, cv2.IMREAD_GRAYSCALE)/255

# # #%%
# # S2 = construct_S(xx,yy,xy)

# #%%
# import os
# import h5py

# out = '/home/brysongray/structure_tensor_analysis/test2_3-18'
# if not os.path.exists(out):
#     os.makedirs(out)
# xx_name = 'M1229-M94--_1_0160_xx.h5'

# with h5py.File(os.path.join(out, xx_name), 'w') as f:
#         f.create_dataset('xx', data=S1[..., 0])
# # cv2.imwrite(os.path.join(out, xx_name), (xx*255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
# # %%
# xx_file = '/home/brysongray/structure_tensor_analysis/test2_3-18/M1229-M94--_1_0160_xx.h5'

# with h5py.File(xx_file, 'r') as f:
#     xx_ = f['xx'][:]

# np.allclose(xx_, S1[..., 0])
# # xx_ = cv2.imread(xx_file, cv2.IMREAD_GRAYSCALE)/255
# # %%
