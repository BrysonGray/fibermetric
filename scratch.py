#%%

import cv2
import matplotlib.pyplot as plt

# theta_file = '/home/brysongray/structure_tensor_analysis/output_images/outputs/cityscape_theta.jp2'
# hsv_file = '/home/brysongray/structure_tensor_analysis/output_images/outputs/cityscape_hsv.jp2'
# hsv_file = '/home/brysongray/structure_tensor_analysis/outputs/M1229-M94--_1_0160_hsv.jp2'
theta_file = '/home/brysongray/structure_tensor_analysis/test_3-18/M1229-M94--_1_0160_theta.png'

# theta = cv2.imread(theta_file, cv2.IMREAD_GRAYSCALE)
theta = cv2.imread(theta_file, cv2.IMREAD_GRAYSCALE)/255

plt.figure(frameon=False, figsize=(20,20))
# plt.imshow(theta, cmap='hsv')
# plt.title('theta')

plt.figure(frameon=False, figsize=(20,20))
plt.imshow(theta, cmap='hsv')
plt.title('theta')

plt.show()

#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from torch.nn.functional import grid_sample
from scipy.ndimage import gaussian_filter, sobel
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
from skimage.transform import resize
import torch
import h5py

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
        with components along the first axis (e.g. x0,x1,x2) and spatial dimensions 
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

def struct_tensor(I, sigma=1, down=0):
    # sobel function gets approx. gradient of image intensity along the specified axis
    print('calculating image gradients...')
    I_y = sobel(I, axis=0)
    I_x = sobel(I, axis=1)

    # construct the structure tensor, s
    print('constructing structure tensors...')
    I_y_sq = gaussian_filter(I_y**2, sigma=sigma)
    I_x_sq = gaussian_filter(I_x**2, sigma=sigma)
    I_xy = gaussian_filter(I_x*I_y, sigma=sigma)

    S = np.stack((I_x_sq, I_xy,
        I_xy, I_y_sq), axis=-1)

    if down:
        S = resize(S, (S.shape[0]//down, S.shape[1]//down, 4), anti_aliasing=True)

    # construct orientation (theta) and anisotropy index (AI)
    print('calculating orientations and anisotropy...')
    d,v = np.linalg.eig(S.reshape((S.shape[0],S.shape[1],2,2)))

    # get only principle eigenvectors
    max_idx = np.abs(d).argmax(axis=-1)
    max_idx = np.ravel(max_idx)
    max_idx = np.array([np.arange(max_idx.shape[0]), max_idx])
    v = np.transpose(v, axes=(0,1,3,2)).reshape(-1,2,2)
    v = v[max_idx[0],max_idx[1]].reshape(S.shape[0],-1,2)
    theta = ((np.arctan(v[...,1] / v[...,0])) + np.pi / 2) / np.pi
    AI = abs(d[...,0] - d[...,1]) / abs(d[...,0] + d[...,1])

    return S, theta, AI

# function for applying rigid transformations to S
def construct_S(xx, yy, xy, A=None):
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
        print('xx shape before interp: ', xx.shape)
        xx = interp(xS, xx[None, ..., None].astype(np.double), AiS).squeeze().numpy()[...,0]
        yy = interp(xS, yy[None, ..., None].astype(np.double), AiS).squeeze().numpy()[...,0]
        xy = interp(xS, xy[None, ..., None].astype(np.double), AiS).squeeze().numpy()[...,0]
        print('xx shape after interp: ', xx.shape)

    S = np.stack((xx, xy, xy, yy), axis=-1)
    S = S.reshape((S.shape[0],S.shape[1],2,2))
    print('S shape: ', S.shape)
    if A is not None:
        # S = np.transpose(np.linalg.inv(A[:2,:2])) @ S @ np.linalg.inv(A[:2,:2])  # transform structure tensors
        S = A[:2,:2] @ S @ np.transpose(A[:2,:2])  # transform structure tensors

    return S

def hsv(theta, AI, I, down=0):
    # make hsv image where hue= primary orientation, saturation= anisotropy, value= original image
    print('constructing hsv image...')
    if down:
        I = resize(I, (I.shape[0]//down, I.shape[1]//down), anti_aliasing=True)
    stack = np.stack([theta,AI,I], -1)
    hsv = matplotlib.colors.hsv_to_rgb(stack)

    return hsv

def plot_quiver(v, theta, density=.125):
    # plot directions as vector field
    plt.figure(frameon=False, figsize=(20,20))
    v_ = resize(v, (v.shape[0]*density, v.shape[1]*density, v.shape[2]), anti_aliasing=True)
    theta_ = resize(theta, (theta.shape[0]*density, theta.shape[1]*density), anti_aliasing=True)
    x,y = np.meshgrid(np.arange(v_.shape[1]), np.arange(v_.shape[0])[::-1])
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    plt.quiver(x, y, v_[...,1], v_[...,0], angles='xy', scale_units='xy', scale=1, headwidth=0.01, headlength=0.01)
    plt.imshow(theta, cmap='hsv', alpha=0.3, interpolation='bilinear', extent=extent)
    plt.show()

#%%
xx_file = '/home/brysongray/structure_tensor_analysis/test_3-18/M1229-M94--_1_0160_xx.h5'
yy_file = '/home/brysongray/structure_tensor_analysis/test_3-18/M1229-M94--_1_0160_yy.h5'
xy_file = '/home/brysongray/structure_tensor_analysis/test_3-18/M1229-M94--_1_0160_xy.h5'
with h5py.File(xx_file, 'r') as f:
    xx = f['xx'][:]
with h5py.File(yy_file, 'r') as f:
    xy = f['yy'][:]
with h5py.File(xy_file, 'r') as f:
    yy = f['xy'][:]

#%%
A90 = np.array([[0, -1, 0],[1, 0, 0], [0, 0, 1]]) # +90 degree rotation in xy plane
A270 = np.array([[0, 1, 0],[-1, 0, 0], [0, 0, 1]]) # -90 degree rotation in xy plane
S = construct_S(xx,yy,xy)
S90 = construct_S(xx,yy,xy,A90)
S270 = construct_S(xx,yy,xy,A270)

d,v = np.linalg.eig(S)
# get only principle eigenvectors
max_idx = np.abs(d).argmax(axis=-1)
max_idx = np.ravel(max_idx)
max_idx = np.array([np.arange(max_idx.shape[0]), max_idx])
v = np.transpose(v, axes=(0,1,3,2)).reshape(-1,2,2)
v = v[max_idx[0],max_idx[1]].reshape(S.shape[0],-1,2)
theta = ((np.arctan(v[...,1] / v[...,0])) + np.pi / 2) / np.pi


d90,v90 = np.linalg.eig(S90)
# get only principle eigenvectors
max_idx90 = np.abs(d90).argmax(axis=-1)
max_idx90 = np.ravel(max_idx90)
max_idx90 = np.array([np.arange(max_idx90.shape[0]), max_idx90])
v90 = np.transpose(v90, axes=(0,1,3,2)).reshape(-1,2,2)
v90 = v90[max_idx90[0],max_idx90[1]].reshape(S90.shape[0],-1,2)
theta90 = ((np.arctan(v90[...,1] / v90[...,0])) + np.pi / 2) / np.pi

d270,v270 = np.linalg.eig(S270)
# get only principle eigenvectors
max_idx270 = np.abs(d270).argmax(axis=-1)
max_idx270 = np.ravel(max_idx270)
max_idx270 = np.array([np.arange(max_idx270.shape[0]), max_idx270])
v270 = np.transpose(v270, axes=(0,1,3,2)).reshape(-1,2,2)
v270 = v270[max_idx270[0],max_idx270[1]].reshape(S270.shape[0],-1,2)
theta270 = ((np.arctan(v270[...,1] / v270[...,0])) + np.pi / 2) / np.pi
#%%
print('calculating orientations and anisotropy...')
d,v = np.linalg.eig(S)

# get only principle eigenvectors
max_idx = np.abs(d).argmax(axis=-1)
max_idx = np.ravel(max_idx)
max_idx = np.array([np.arange(max_idx.shape[0]), max_idx])
v = np.transpose(v, axes=(0,1,3,2)).reshape(-1,2,2)
v = v[max_idx[0],max_idx[1]].reshape(S.shape[0],-1,2)
theta = ((np.arctan(v[...,1] / v[...,0])) + np.pi / 2) / np.pi

#%%
plot_quiver(v, theta, density=.125)
plot_quiver(v90, theta90, density=.125)
plot_quiver(v270, theta270, density=.125)
#%%
plt.figure(figsize=(10,10))
plt.imshow(theta, cmap='hsv')
plt.title('theta')

plt.show()

#%%
I_file = '/home/brysongray/data/m1229/m1229_other_data/M1229-M94--_1_0160.jp2'
down=32
I = load_img(I_file, img_down=down,reverse_intensity=True)

#%%
print('calculating orientations and anisotropy...')
d,v = np.linalg.eig(S90)

# get only principle eigenvectors
max_idx = np.abs(d).argmax(axis=-1)
max_idx = np.ravel(max_idx)
max_idx = np.array([np.arange(max_idx.shape[0]), max_idx])
v = np.transpose(v, axes=(0,1,3,2)).reshape(-1,2,2)
v = v[max_idx[0],max_idx[1]].reshape(S90.shape[0],-1,2)
theta = ((np.arctan(v[...,1] / v[...,0])) + np.pi / 2) / np.pi
# AI = abs(d[...,0] - d[...,1]) / abs(d[...,0] + d[...,1])

plt.figure(figsize=(10,10))
plt.imshow(theta, cmap='hsv')
plt.title('theta')

plt.show()

# %%

# img = 'radial.jpeg'
# img = './input_images/cityscape.jpg'
# img = '/home/brysongray/data/m1229/m1229_other_data/M1229-M94--_1_0160.jp2'
# # img = '/home/brysongray/data/MD847/MD847-My10-2021.12.13-22.39.49_MD847_1_0028_lossy.jp2'
# # img = '/home/brysongray/data/MD847/MD847-My10-2021.12.13-22.39.49_MD847_3_0030_lossy.jp2'
# # img = '/home/brysongray/data/MD847/MD847-My100-2021.12.14-10.51.54_MD847_1_0298_lossy.jp2'
# # img = '/home/brysongray/data/MD847/MD847-My101-2021.12.14-10.59.05_MD847_1_0301_lossy.jp2'
# # img = '/home/brysongray/data/MD847/MD847-My102-2021.12.14-11.07.43_MD847_1_0304_lossy.jp2'

# I = load_img(img, reverse_intensity=True)
# # S, theta, AI = struct_tensor(I, down=32)
# # hsv_ = hsv(theta, AI, I, down=32)

#%%

# # plt.figure(figsize=(10,10))
# # plt.imshow(theta, cmap='hsv')
# # plt.title('theta')

# # plt.figure(figsize=(10,10))
# # plt.imshow(AI, cmap='gray')
# # plt.title('Anisotropy Index')

# plt.figure(frameon=False, figsize=(20,20))
# # x,y = np.meshgrid(np.arange(v.shape[1]), np.arange(v.shape[0]))
# # extent = np.min(x), np.max(x), np.max(y), np.min(y)
# plt.imshow(hsv_)
# # plt.quiver(x, y, v[...,1], v[...,0])
# # plt.imshow(I, cmap='gray', alpha=0.5, interpolation='bilinear', extent=extent)
# plt.title('hsv')

# plt.show()
#%%
# import cv2
# import matplotlib.pyplot as plt

# img = '/home/brysongray/structure_tensor_analysis/outputs/cityscape_hsv.jp2'

# hsv_ = cv2.imread(img)

# plt.figure(frameon=False, figsize=(20,20))

# plt.imshow(hsv_)
# plt.title('hsv')

# plt.show()

# %%
import emlddmm
import numpy as np

id_file = '/home/brysongray/data/MD816_mini/HR_NIHxCSHL_50um_14T_M1_masked.vtk'
disp_file = '/home/brysongray/emlddmm/outputs/transformation_graph_4-6_outputs/MRI/CCF_to_MRI/transforms/MRI_to_CCF_displacement.vtk'

xI,I,_,_ = emlddmm.read_data(id_file)
xdisp,disp,_,_ = emlddmm.read_data(disp_file)

XI = np.stack(np.meshgrid(xI[0],xI[1],xI[2], indexing='ij'))
phi = disp[0] + XI 

print(phi.shape)
# %%
Xin = np.stack(np.meshgrid(tuple(xI)))
print(Xin.shape)
# %%
import nibabel as nib

T_file = '/home/brysongray/structure_tensor_analysis/dki/dki_tensor.nii.gz'
T_ = nib.load(T_file)

T = T_.get_fdata()

T_head = T_.header
dim = T_head['dim'][1:4]
pixdim = T_head['pixdim'][1:4]

xT = []
for i in range(len(dim)):
    x = (np.arange(dim[i]) - dim[i]/2) * pixdim[i]
    xT.append(x)

print([x.shape for x in xT])



# %%
import numpy as np

# dummy tensor field
# make tensor pointing vertically
up = np.array([[10,0,0],
              [0,1,0],
              [0,0,1]])

# make tensor pointing horizontally
right = np.array([[1,0,0],
              [0,10,0],
              [0,0,1]])

# identity
id = np.array([[1,0,0],
               [0,1,0],
               [0,0,1]])

# initialize tensor field
T = np.zeros((10,10,3,3))

T[2:5, 2:8, ...] = up
T[5:8, 2:8, ...] = right
T[:2,...] = id # pad the edges with the identity matrix
T[8:,...] = id
T[:,:2,...] = id
T[:,8:,...] = id


# create grid
xI = []
for i in range(3):
    x = np.arange(T.shape[i]) - (T.shape[i]-1)/2
    xI.append(x)
X = np.stack(np.meshgrid(xI[0],xI[1],xI[2]),axis=-1)

# rotate grid 45 degrees about z axis
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta), np.cos(theta), 0],
              [0,0,1]])

Xrot = (R @ X[...,None])[...,0]

# %%

# now we have to draw ellipses
# first draw a square
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.linalg import sqrtm
f = plt.figure()
ax = f.add_subplot(projection='3d')

n0 = 10
n1 = 10
t0 = np.arange(n0+1)/n0 * 2.0*np.pi # alpha
t1 = np.arange(n1+1)/n1 * np.pi # polar
T0,T1 = np.meshgrid(t0,t1,indexing='ij')
Y0 = np.sin(T1)*np.cos(T0)
Y1 = np.sin(T1)*np.sin(T0)
Y2 = np.cos(T1)
ax.plot_surface(Y0,Y1,Y2,shade=False,color='r',edgecolor='k')

#%%
J = T
w,v = np.linalg.eigh(J)
trace = np.sum(w,-1)

R = J/np.trace(J,axis1=-1,axis2=-2)[...,None,None]
FA = np.sqrt(0.5*(3.0 - 1.0/np.trace(R@R,axis1=-1,axis2=-2)))
FA[np.isnan(FA)] = 0


#%%
from matplotlib import cm
Jd = J
FAd = FA
traced = trace
vd = v

f = plt.figure(figsize=(8,8))
ax = f.add_subplot(projection='3d')
ax.view_init(elev=90,azim=90)
#ax.view_init(elev=80,azim=180)
# x0 = np.arange(traced.shape[0])
# x1 = np.arange(traced.shape[1])
# # X0,X1 = np.meshgrid(x0,x1,indexing='ij')

# ax.plot_surface(X0,X1,traced,cmap=cm.gray) # don't know how to show this
d = 1
for i in range(0,Jd.shape[0],d):
    for j in range(0,Jd.shape[1],d):
        A = Jd[i,j]
        A12 = sqrtm(A)
        Z0 = A12[0,0]*Y0 + A12[0,1]*Y1 + A12[0,2]*Y2
        Z1 = A12[1,0]*Y0 + A12[1,1]*Y1 + A12[1,2]*Y2
        Z2 = A12[2,0]*Y0 + A12[2,1]*Y1 + A12[2,2]*Y2
        scale = 0.125#*FAd[i,j]        
        ax.plot_surface(scale*Z0+i,scale*Z1+j,.0125*Z2+0.5,shade=False,color=np.abs(vd[i,j,:,-1]),edgecolor=None)
        ax.set_zlim(0,1)

#%%
# # plot dti with glyphs

# R = J/np.trace(J,axis1=-1,axis2=-2)[...,None,None]
# FA = np.sqrt(0.5*(3.0 - 1.0/np.trace(R@R,axis1=-1,axis2=-2)))
# #f,ax = plt.subplots()
# #ax.imshow(FA,origin='upper')

# from matplotlib import cm
# w,v = np.linalg.eigh(J,)
# trace = np.sum(w,-1)
# f = plt.figure(figsize=(8,8))
# ax = f.add_subplot(projection='3d')
# ax.view_init(elev=90,azim=180)
# # ax.view_init(elev=80,azim=180)
# x0 = np.arange(J.shape[0])
# x1 = np.arange(J.shape[1])
# X0,X1 = np.meshgrid(x0,x1,indexing='ij')
# # cut off first 10 pixels
# r0 = 20
# r1 = -10
# ax.plot_surface(X0[r0:r1],X1[r0:r1],trace[r0:r1],cmap=cm.gray) # don't know how to show this
# for i in range(10,J.shape[0]+r1,6//2):
#     for j in range(0,J.shape[1],6//2):
#         A = J[i,j]
#         A12 = sqrtm(A)
#         Z0 = A12[0,0]*Y0 + A12[0,1]*Y1 + A12[0,2]*Y2
#         Z1 = A12[1,0]*Y0 + A12[1,1]*Y1 + A12[1,2]*Y2
#         Z2 = A12[2,0]*Y0 + A12[2,1]*Y1 + A12[2,2]*Y2
#         scale = 100*FA[i,j]
#         #scale = 1.0/(np.sqrt(w[i,j,-1]) + 1e-3)
#         ax.plot_surface(scale*Z2+i,scale*Z0+j,scale*Z1+1,shade=False,color=np.abs(v[i,j,:,-1]),edgecolor=None)
        
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# ax.set_zlim(ax.get_xlim())
# # f.savefig('ch9-ellipses-1.png')
# %%
import numpy as np
import nibabel as nib
import emlddmm
from emlddmm import read_data, interp
import matplotlib.pyplot as plt
#%%

def read_dti(dti_path):
    T_ = nib.load(dti_path)
    # get tensor data
    T = T_.get_fdata()
    # # get tensor data coordinates
    T_head = T_.header
    dim = T_head['dim'][1:4]
    pixdim = T_head['pixdim'][1:4]

    xT = []
    for i in range(3):
        x = (np.arange(dim[i]) - (dim[i]-1)/2) * pixdim[i]
        xT.append(x)
    
    # Ts = []
    # for i in range(6):
    #     # Resample components of the tensor field at transformed points
    #     x = interp(xT, T[...,i][None], X)
    #     Ts.append(x)

    # T has the 6 unique elements of the symmetric positive-definite diffusion tensors.
    # The lower triangle of the matrix must be filled and last 2 dimensions reformed into 3x3 tensors.
    T = np.stack((T[...,0],T[...,3],T[...,4],T[...,3],T[...,1],T[...,5],T[...,4],T[...,5],T[...,2]), axis=-1)
    T = T.reshape(T.shape[:-1]+(3,3)) # result is row x col x slice x 3x3
    
    return xT, T

def visualize(T,xT, fig=None):
    """ Visualize DTI

    Visualize diffusion tensor images with RGB encoded tensor orientations.
    
    Parameters
    ----------
    T: numpy array
        Diffusion tensor volume with the last two dimensions being 3x3 containing tensor components.
    """
    # Get tensor principle eigenvectors
    w, e = np.linalg.eigh(T)
    princ_eig = e[...,-1]
    # scale values from 0 to 1
    princ_eig = (princ_eig - np.min(princ_eig)) / (np.max(princ_eig) - np.min(princ_eig))
    # transpose to C x nslice x nrow x ncol
    princ_eig = princ_eig.transpose(3,0,1,2)
    # visualize
    emlddmm.draw(princ_eig, xJ=xT, fig=fig, n_slices=7)

    return princ_eig
#%%

dti_path = '/home/brysongray/structure_tensor_analysis/dki/dki_tensor.nii.gz'

# T_ = nib.load(dti_path)
# # get tensor data
# T = T_.get_fdata()

xT, T = read_dti(dti_path)

fig = plt.figure(figsize=(10,10))
eigs = visualize(T,xT,fig=fig)
emlddmm.write_vtk_data('dki_tensor_directions.vtk', xT, eigs, 'dki_tensor_directions')
# w, e = np.linalg.eigh(T)
# princ_eig = e[...,-1]
# %%
img_path = '/home/brysongray/structure_tensor_analysis/dki_tensor_directions.vtk'

img = emlddmm.read_vtk_data(img_path)
# %%
