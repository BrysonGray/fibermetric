#%%
import numpy as np
import emlddmm
from emlddmm import read_data, interp
import dti
import matplotlib.pyplot as plt
import os
import h5py
import torch
# from sklearn.cluster import KMeans
# import nibabel as nib

jacobian = lambda X,dv : np.stack(np.gradient(X, dv[0],dv[1],dv[2], axis=(0,1,2)), axis=-1)


# %%
# TRANSFORM DTI to HIST REGISTERED SPACE
#################################################################################################################################################################################
# dwi_to_mri_path = '/home/brysongray/emlddmm/amyg_maps_for_bryson/mri_avg_dwi_to_mri.vtk'
# mri_to_hist_path = '/home/brysongray/data/human_amyg/amyg_maps_for_bryson/mri_to_registered_histology_displacement.vtk'
hist_to_mri_path = '/home/brysongray/data/human_amyg/amyg_maps_for_bryson/registered_histology_to_mri_displacement.vtk'
dti_path = '/home/brysongray/data/dki/dki_tensor.nii.gz'
hist_path = '/home/brysongray/data/human_amyg/amyg_maps_for_bryson/mri_avg_b0_to_registered_histology.vtk'

# load disp and original image
_,hist_to_mri,_,_ = read_data(hist_to_mri_path)
hist_to_mri_T = np.stack((hist_to_mri[0][1], hist_to_mri[0][0], hist_to_mri[0][2])).transpose(0,2,1,3)
xI,I,title,_ = read_data(hist_path)
#%%
# match axes to dti
# I_T = I.transpose(0,2,1,3)
# xI_T = [xI[1],xI[0],xI[2]]
dx = [(x[1] - x[0]) for x in xI]
# get transformed coordinates
Xin = np.stack(np.meshgrid(xI[0],xI[1],xI[2], indexing='ij')) # stack coordinates to match dti
X = hist_to_mri[0] + Xin

# read dti
xT, T = dti.read_dti(dti_path)
# transpose and switch components to match histology space orientation
T = np.stack((T[...,0,0],T[...,1,1],T[...,2,2],T[...,0,1],T[...,0,2],T[...,1,2]),-1)
T = np.stack((T[...,1],T[...,3],T[...,5],T[...,3],T[...,0],T[...,4],T[...,5],T[...,4],T[...,2]), axis=-1).transpose(1,0,2,3)
T = T.reshape(T.shape[:3]+(3,3))
xT = [xT[1], xT[0], xT[2]]
# resample dti in registered histology space
Tnew = dti.interp_dti(T,xT,X)
# rotate tensors
J = jacobian(X.transpose(1,2,3,0),dx)
Q = dti.ppd(Tnew,J)
Tnew = Q @ Tnew @ Q.transpose(0,1,2,4,3)

#%%
# visualize transformed dti
fig1 = plt.figure(figsize=(12,12))
dti_to_hist = dti.visualize(Tnew,xI,fig=fig1)

# visualize avg dwi in registered histology space
fig2 = plt.figure(figsize=(12,12))
emlddmm.draw(I,xI,fig2)

# visualize original dti
fig3 = plt.figure(figsize=(12,12))
T_img = dti.visualize(T,xT,fig=fig3)

# %%
# save transformed tensors
T_name = 'dti_to_registered_histology.h5'
out = 'output_images/'
with h5py.File(os.path.join(out, T_name), 'w') as f:
        f.create_dataset('DTI transformed', data=Tnew)

# %%
# TRANSFORM structure tensor images to HIST REGISTERED SPACE
#################################################################################################################################################################################

# load structure tensors
st_path = '/home/brysongray/data/human_amyg/sta_output'
nissl_myelin_rigid = '/home/brysongray/data/human_amyg/nissl_myelin_rigid'
# get only slices that match numbers on rigid transforms
# list of slice numbers
transform_paths = [x for x in os.listdir(nissl_myelin_rigid) if 'npz' in x] # get paths
transform_paths = sorted(transform_paths, key=lambda x: int(x.split('_')[-4])) # sort paths
transform_nums = [x.split('_')[-4] for x in transform_paths]

# intersection of lists
st_list = [x for x in os.listdir(st_path) if 'h5' in x]
st_list = sorted(st_list, key=lambda x: x.split('_')[-3])
st_list_nums = [x.split('_')[-3] for x in st_list]
ind_dict = dict((k,i) for i,k in enumerate(st_list_nums))
inter = sorted(list(set(ind_dict).intersection(transform_nums)))
indices = [ind_dict[x] for x in list(inter)]
st_list_x = np.array(st_list)[indices]

# read tensors
S_ = []
nS_ = np.zeros((len(st_list_x),3), dtype=int)
dS = np.array([10.0,14.72,14.72])
for i in range(len(st_list_x)):
    fname = st_list_x[i]
    with h5py.File(os.path.join(st_path,fname),'r') as f:
        img = f[list(f.keys())[0]][:]
    # combine 2x2 structure tensors into one dimension (xx,xy,yy)
    img = np.stack((img[...,0,0], img[...,0,1], img[...,1,1]),axis=-1)
    nS_[i] = np.array(img.shape)
    S_.append(img)

# pad images
nSm = np.max(nS_,0)
nSm = (np.quantile(nS_,0.95,axis=0)*1.01).astype(int) 
print('padding and assembling into 3D volume')
S = np.zeros((len(S_),nSm[0],nSm[1],3), dtype=np.float32)
W0 = np.zeros((len(S_),nSm[0],nSm[1]))
for i in range(len(S_)):
    S__ = S_[i]
    topad = nSm - np.array(nS_[i])
    # just pad on the left and I'll fill it in
    topad = np.array(((topad[0]//2,0),(topad[1]//2,0),(0,0)))
    # if there are any negative values I need to crop, I'll just crop on the right
    if np.any(np.array(topad)<0):
        if topad[0][0] < 0:
            S__ = S__[:S.shape[1]]
            topad[0][0] = 0
        if topad[1][0] < 0:
            S__ = S__[:,:S.shape[2]]
            topad[1][0] = 0
    Sp = np.pad(S__,topad,constant_values=np.nan)
    W0_ = np.logical_not(np.isnan(Sp[...,0]))
    Sp[np.isnan(Sp)] = 0
    W0[i,:W0_.shape[0],:W0_.shape[1]] = W0_
    S[i,:W0_.shape[0],:W0_.shape[1],:] = Sp
S = np.transpose(S,(3,0,1,2))    
nS = np.array(S.shape)
xS = [np.arange(n)*d - (n-1)*d/2.0 for n,d in zip(nS[1:],dS)]
# W0 = W0 * np.logical_not(np.all(S==0.0,0))
# W0 = W0 * np.logical_not(np.all(S==1.0,0))
del S_, S__, Sp, nS_, nS, W0, img
#%%
# FIRST REGISTER TO NISSL INPUT SPACE
# load rigid transforms
A2d = []
for path in transform_paths:
    A2d_ = np.load(os.path.join(nissl_myelin_rigid,path))['arr_0']
    A2d.append(A2d_)
xS = [torch.tensor(x) for x in xS]
XS = torch.stack(torch.meshgrid(xS, indexing='ij'),-1)
# apply inverse of nissl to myelin affine transforms
A2d = torch.as_tensor(np.stack(A2d))#,dtype=dtype,device=device)
A2di = torch.inverse(A2d)
points = (A2di[:, None, None, :2, :2] @ XS[..., 1:, None])[..., 0] 
m0 = torch.min(points[..., 0])
M0 = torch.max(points[..., 0])
m1 = torch.min(points[..., 1])
M1 = torch.max(points[..., 1])
# # construct a recon domain
# dJ = [x[1] - x[0] for x in x_series]
# print('dJ shape: ', [x.shape for x in dJ])
xr0 = torch.arange(float(m0), float(M0), dS[1], device=m0.device, dtype=m0.dtype)
xr1 = torch.arange(float(m1), float(M1), dS[2], device=m0.device, dtype=m0.dtype)
xr = xS[0], xr0, xr1
XR = torch.stack(torch.meshgrid(xr, indexing='ij'), -1).float()
del XS, points
# reconstruct 2d series
A2d = A2d.float()
XR[..., 1:] = (A2d[:, None, None, :2, :2] @ XR[..., 1:, None])[..., 0] + A2d[:, None, None, :2, -1]
#%%
XR = XR.permute(3, 0, 1, 2)
xS = [x.float() for x in xS]
Sr = emlddmm.interp(xS, S, XR)
del S, XR
#%%
# stack component into tensors
Sr = torch.stack((Sr[0], Sr[1], Sr[1], Sr[2]), dim=-1)
Sr = Sr.reshape(Sr.shape[:3] + (2,2)).detach().numpy()
#%%
# write out registered structure tensors
st_name = 'structure_tensors_to_registered_histology.h5'
out = 'output_images/'
with h5py.File(os.path.join(out, st_name), 'w') as f:
        f.create_dataset('sta registered', data=Sr)
#%%
# get principle orientations
w,e = np.linalg.eigh(Sr)
e_prime = e[...,-1]

#%%
# load dti_to_registered_histology
T_path = '/home/brysongray/structure_tensor_analysis/output_images/dti_to_registered_histology.h5'

T_ = h5py.File(T_path, 'r')
T = np.array(T_[list(T_.keys())[0]])

# Get FA from tensors
# Get tensor principle eigenvectors
w, e = np.linalg.eigh(T)
princ_eig = e[...,-1]
trace = np.trace(T, axis1=-2, axis2=-1)
# transpose to C x nslice x nrow x ncol
w = w.transpose(3,0,1,2)
FA = np.sqrt((3/2) * (np.sum((w - (1/3)*trace)**2,axis=0) / np.sum(w**2, axis=0)))
FA = np.nan_to_num(FA)
FA = FA/np.max(FA)

fig = plt.figure(figsize=(12,12))
plt.imshow(FA[108,...])

#%%
# create mask based on FA threshold
FA_thresh = 0.125
FA_mask = np.where(FA >= FA_thresh, 1.0, 0.0)


fig = plt.figure(figsize=(12,12))
plt.imshow(FA_mask[108,...])

# mask based on out-of-slice angle
# slices correspond to the first component
theta_thresh = np.pi/3
theta = np.abs(np.arcsin(princ_eig[...,0]))
theta_mask = np.where(theta <= theta_thresh, 1.0, 0.0)

# mask out brain from T2 image
brain_thresh = 900
brain_mask = np.where(I[0] >= brain_thresh, 1.0, 0.0)

fig = plt.figure(figsize=(12,12))
plt.imshow(theta_mask[108,...])

WM_mask = FA_mask * theta_mask * brain_mask
GM_mask = brain_mask - WM_mask
fig = plt.figure(figsize=(12,12))
plt.imshow(WM_mask[108])
fig = plt.figure(figsize=(12,12))
plt.imshow(GM_mask[108])
# try using K-means to create mask

# kmean_FA = KMeans(n_clusters=3).fit(np.ravel(FA)[...,None])
# FA_labels = np.reshape(kmean_FA.labels_, FA.shape) / 2

# save out
emlddmm.write_vtk_data('output_images/WM_mask.vtk', xI, WM_mask[None], title='WM_in_registered_hist_space')
emlddmm.write_vtk_data('output_images/GM_mask.vtk', xI, GM_mask[None], title='GM_in_registered_hist_space')
emlddmm.write_vtk_data('output_images/brain_mask.vtk', xI, brain_mask[None], title='brain_mask_in_registered_hist_space')

# emlddmm.write_vtk_data('output_images/FA.vtk', xI, FA_labels[None], title='FA_in_registered_hist_space')
# FA_img = nib.Nifti1Image(FA_labels, affine=np.eye(4))
# nib.save(FA_img,'output_images/FA.nii')
#%%

# Apply mask to dti

# Project tensors into slice plane

# Compute angle between in-place dti tensor and structure tensor

# statistical analysis


# %%
# TRANSFORM FROM HIST REGISTERED TO HIST INPUT SPACE
#################################################################################################################################################################################
# import torch
# import emlddmm
# import numpy as np

# dtype = torch.float
# device = 'cpu'

# src_path = '/home/brysongray/structure_tensor_analysis/output_images/dti_to_registered_histology.vtk'
# transforms = '/home/brysongray/data/human_amyg/amygdala_outputs_v00/input_histology/registered_histology_to_input_histology/transforms/'

# xJ, J, J_title, _ = emlddmm.read_data(src_path) # the image to be transformed
# J = J.astype(float)
# J = torch.as_tensor(J,dtype=dtype,device=device)
# xJ = [torch.as_tensor(np.copy(x),dtype=dtype,device=device) for x in xJ]

# x_series = xJ
# X_series = torch.stack(torch.meshgrid(x_series, indexing='ij'), -1)

# transforms_ls = sorted(os.listdir(transforms), key=lambda x: x.split('_matrix.txt')[0][-4:])
# #%%
# A2d = []
# for t in transforms_ls:
#     A2d_ = np.genfromtxt(os.path.join(transforms, t), delimiter=' ')
#     # note that there are nans at the end if I have commas at the end
#     if np.isnan(A2d_[0, -1]):
#         A2d_ = A2d_[:, :A2d_.shape[1] - 1]
#     A2d.append(A2d_)

# A2d = torch.as_tensor(np.stack(A2d),dtype=dtype,device=device)
# A2di = torch.inverse(A2d)
# points = (A2di[:, None, None, :2, :2] @ X_series[..., 1:, None])[..., 0] 
# m0 = torch.min(points[..., 0])
# M0 = torch.max(points[..., 0])
# m1 = torch.min(points[..., 1])
# M1 = torch.max(points[..., 1])
# # construct a recon domain
# dJ = [x[1] - x[0] for x in x_series]
# # print('dJ shape: ', [x.shape for x in dJ])
# xr0 = torch.arange(float(m0), float(M0), dJ[1], device=m0.device, dtype=m0.dtype)
# xr1 = torch.arange(float(m1), float(M1), dJ[2], device=m0.device, dtype=m0.dtype)
# xr = x_series[0], xr0, xr1
# XR = torch.stack(torch.meshgrid(xr), -1)
# # reconstruct 2d series
# Xs = torch.clone(XR)
# Xs[..., 1:] = (A2d[:, None, None, :2, :2] @ XR[..., 1:, None])[..., 0] + A2d[:, None, None, :2, -1]
# Xs = Xs.permute(3, 0, 1, 2)
# Jr = dti.interp_dti(xJ, J, Xs)
