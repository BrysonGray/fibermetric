import numpy as np
import nibabel as nib
import argparse
from utils import read_data, interp, draw
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
from dipy.core.sphere import Sphere
from dipy.reconst.shm import sh_to_sf
from tqdm import tqdm
import sympy
from sympy import Ynm, Symbol, integrate
import pickle

#############################
# diffusion tensor functions
#############################

# preservation of principle directions
def ppd(tensors,J):
    """ Preservation of Principle Directions

    Transforms tensors, given a Jacobian field, using preservation of principle directions method.

    Parameters
    ----------
    tensors: numpy array
        5 dimensional array with the last two dimensions being 3x3 containing tensor components.
    J: numpy array
        5 dimensional array with the last two dimensions being 3x3 containing Jacobian matrix components.

    """
    # define function to construct rotation matrix from axis of rotation and angle
    rot = lambda n, theta : np.array([[np.cos(theta)+n[...,0,None]**2*(1-np.cos(theta)), n[...,0,None]*n[...,1,None]*(1-np.cos(theta))-n[...,2,None]*np.sin(theta), n[...,0,None]*n[...,2,None]*(1-np.cos(theta))+n[...,1,None]*np.sin(theta)],
                                    [n[...,0,None]*n[...,1,None]*(1-np.cos(theta))+n[...,2,None]*np.sin(theta), np.cos(theta)+n[...,1,None]**2*(1-np.cos(theta)), n[...,1,None]*n[...,2,None]*(1-np.cos(theta))-n[...,0,None]*np.sin(theta)],
                                    [n[...,0,None]*n[...,2,None]*(1-np.cos(theta))-n[...,1,None]*np.sin(theta), n[...,1,None]*n[...,2,None]*(1-np.cos(theta))+n[...,0,None]*np.sin(theta), np.cos(theta)+n[...,2,None]**2*(1-np.cos(theta))]]).squeeze().transpose(2,3,4,0,1)

    # compute unit eigenvectors, e, of tensors
    w,e = np.linalg.eigh(tensors)
    e1 = e[...,-1]
    e2 = e[...,-2]
    # compute unit vectors n1 and n2 in the directions of J@e1 and J@e2
    Je1 = np.squeeze(J @ e1[...,None])
    n1 = Je1 / np.linalg.norm(Je1, axis=-1)[...,None]
    Je2 = np.squeeze(J @ e2[...,None])
    n2 = Je2 / np.linalg.norm(Je2, axis=-1)[...,None]
    # compute a rotation matrix, R1, that maps e1 onto n1
    theta = np.arccos(np.squeeze(e1[..., None, :] @ n1[..., None]))[...,None]
    r = np.cross(e1,n1) / np.sin(theta)
    # r will be nan wherever e1 and n1 align, i.e. where Je1 == e1. NOTE: Divide by zero warning is suppressed
    # replace r whereever there are nans with a different r. I use n2 instead of n1.
    theta2 = np.arccos(np.squeeze(e1[..., None, :] @ n2[..., None]))[...,None]
    r2 = np.cross(e1,n2) / np.sin(theta2)
    r[np.isnan(r)] = r2[np.isnan(r)]
    # theta[np.where(theta==0)] = theta2[np.where(theta==0)]
    R1 = rot(r,theta)
    # compute a secondary rotation, about n1, to map e2 from its position after the first rotation, R1 @ e2,
    # to the n1-n2 plane.
    Pn2 = n2 - (n2[..., None, :] @ n1[..., None])[...,0] * n1
    Pn2 = Pn2 / np.linalg.norm(Pn2, axis=-1)[...,None]
    R1e1 = np.squeeze(R1 @ e1[...,None])
    R1e2 = np.squeeze(R1 @ e2[...,None])
    phi = np.arccos(np.squeeze(R1e2[..., None, :] @ Pn2[..., None]) / (np.linalg.norm(R1e2) * np.linalg.norm(Pn2)))[...,None]
    R2 = rot(R1e1, phi)

    Q = R2 @ R1

    return Q
    

# TODO: finite strain
def fs(J):
    Q = J / (J @ np.transpose(J, axes=(0,1,2,4,3)))**(-1/2)

    return Q


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


def interp_dti(T, xT, X):
    """ Interpolate DTI

    """
    if len(T.shape) == 5: # assume 3x3 tensors along last dimensions
        T = np.stack((T[...,0,0],T[...,1,1],T[...,2,2],T[...,0,1],T[...,0,2],T[...,1,2]),-1)
    else:
        raise Exception("T must contain 3x3 tensors or 6 diffusion components in last dimension") 
    Tnew = []
    for i in range(6):
        # Resample components of the tensor field at transformed points
        x = interp(xT, T[...,i][None], X)
        Tnew.append(x)
    
    Tnew = np.stack((Tnew[0], Tnew[3], Tnew[4], Tnew[3], Tnew[1], Tnew[5], Tnew[4], Tnew[5], Tnew[2]), axis=-1)
    Tnew = Tnew.reshape(Tnew.shape[:-1]+(3,3)).squeeze() # result is row x col x slice x 3x3

    return Tnew


def visualize(T,xT, **kwargs):
    """ Visualize DTI

    Visualize diffusion tensor images with RGB encoded tensor orientations.
    
    Parameters
    ----------
    T : numpy array
        Diffusion tensor volume with the last two dimensions being 3x3 containing tensor components.
    xT : list
        A list of 3 numpy arrays.  xT[i] contains the positions of voxels
        along axis i.  Note these are assumed to be uniformly spaced. The default
        is voxels of size 1.0.
    kwargs : dict
        Other keywords will be passed on to the draw function. For example
        include cmap='gray' for a gray colormap
    
    Returns
    -------
    img : numpy array
        4 dimensional array with the first dimension containing color channel components corresponding
        to principal eigenvector orientations.
    """
    # Get tensor principle eigenvectors
    w, e = np.linalg.eigh(T)
    princ_eig = e[...,-1]
    # transpose to C x nslice x nrow x ncol
    img = np.abs(princ_eig).transpose(3,0,1,2)
    trace = np.trace(T, axis1=-2, axis2=-1)
    w = w.transpose(3,0,1,2)
    FA = np.sqrt((3/2) * (np.sum((w - (1/3)*trace)**2,axis=0) / np.sum(w**2, axis=0)))
    FA = np.nan_to_num(FA)
    FA = FA/np.max(FA)
    # scale img by FA
    img = img * FA
    # visualize
    draw(img, xJ=xT, **kwargs)

    return img

####################
# ODF functions
####################

def sh_to_cf_numeric(sh_signal, ndir, nbins, norm=True):
    
    """
    integration: add up the values of the function in each bin (holding phi constant),
    then multiply by the difference between boundaries (pi - 0 = pi)
    and divide by N-1 where N is the number of samples.

    Parameters
    ----------
    sh_signal : numpy array
        array of spherical harmonic coefficients.
        The first dimension must be coefficients, followed by spatial dimensions.
    ndir : int
        number of directions in theta from which to sample for integrating
    nbins : int
        number of bins for the function output
    scale : bool
        scales the circular functions to the interval [0,1].
    Returns
    -------
    cf : numpy array
        array of discrete functions on a circle with the first dimension being
        number of bins, followed by the shape of the input spatial dimensions.
    """
    
    step = 2*np.pi/nbins
    bins = np.arange(0,2*np.pi,step)
    cf = []
    # compute angles at which to sample
    phi = bins[None].T*np.ones((nbins, ndir))
    theta = np.arange(0, np.pi, np.pi/ndir) # uniformly spaced numbers from 0 to pi
    sh_flat = sh_signal.reshape((sh_signal.shape[0],-1)) # flatten the image (Ncoeffs, Nslices x Nrows x Ncols)
    dy = np.sin(theta) * np.pi/(ndir-1)
    cf = np.zeros((nbins,sh_flat.shape[1])) # (nbins, Nslices x Nrows x Ncols)

    for i in tqdm(range(nbins), desc=f'integrating...'):
        # sample from the spherical harmonic function
        x = Sphere(theta=theta, phi=phi[i]) # the input direction needs to be a dipy sphere object
        for j in range(sh_flat.shape[1]): # loop over every voxel
            y = sh_to_sf(sh_flat[:,j], x, sh_order=8) # sample from odf
            cf[i,j] = np.sum(y*dy) # integrate
    cf_shape = list(sh_signal.shape[1:])
    cf_shape.insert(0,nbins)
    cf = np.reshape(cf, tuple(cf_shape))
    if norm:
        cf = cf / np.sum(cf)

    return cf

def sh_to_cf(sh_signal, ndir, source):
    """
    Integrate ODFs as spherical harmonics over theta (polar angle) using symbolic integrals.

    Parameters
    ----------
    sh_signal : numpy array
         Array of spherical harmonic coefficients.
         The first dimension must be coefficients, followed by spatial dimensions.
    ndir : int
        Number of directions in theta from which to sample for integrating
    source : str
        Firectory containing Yphi.p file storing the integrated basis functions.
    
    Returns
    -------
    cf : numpy array
        array of discrete functions on a circle with the first dimension being
        number of bins, followed by the shape of the input spatial dimensions.
    """
    # set up basis (based on Dipy descoteaux07 basis)
    theta = Symbol("theta")
    phi = Symbol("phi")
    degree = {1:0,
             2:6,
             15:4,
             28:6,
             45:8,
             66:10,
             91:12,
             120:14}
    n = degree[sh_signal.shape[0]]
    print(f'spherical harmonic degree is {n}.')
    try:
        path = os.path.join(source,'Yphi.p')
        with open(path, 'rb') as f:
            Yphi = pickle.load(f)
            print(f"found basis saved at {path}")
            if len(Yphi) != sh_signal.shape[0]:
                raise Exception("basis does not match the spherical harmonic signal.")

    except:
        Yphi = []
        print("computing a new basis...")
        for n in tqdm(np.arange(n+1, step=2)):
            for m in np.arange(-n,n+1):
                if m < 0:
                    Yphi.append(sympy.sqrt(2)*sympy.re(integrate(Ynm(n,m,theta,phi).expand(func=True),(theta,0,sympy.pi))))
                elif m == 0:
                    Yphi.append(integrate(Ynm(n,m,theta,phi).expand(func=True),(theta,0,sympy.pi)))
                elif m > 0:
                    Yphi.append(sympy.sqrt(2)*sympy.im(integrate(Ynm(n,m,theta,phi).expand(func=True),(theta,0,sympy.pi))))
        print('done')
        with open(os.path.join(source,'Yphi.p'), 'wb') as f:
            pickle.dump(Yphi, f)
    Y_matrix = np.zeros((ndir,len(Yphi)))
    print(f'Y matrix shape: {Y_matrix.shape}')
    for t in tqdm(range(ndir)):
        for i,y in enumerate(Yphi):
            x = t*2*np.pi/ndir
            Y_matrix[t,i] = float(y.evalf(subs={phi:x}))
    cf = np.moveaxis(np.squeeze(Y_matrix @ np.moveaxis(sh_signal,0,-1)[...,None]),-1,0)

    return cf

def load_odf(path):
    files = os.listdir(path) 
    files.sort()
    sh_list = []
    for i in files:
        if os.path.splitext(i)[1] == '.gz':
            sh_img = nib.load(os.path.join(path, i))
            sh = sh_img.get_fdata()
            sh_list.append(sh)
    I = np.array(sh_list)

    return I

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("tensors", help="diffusion tensor field in nii format")
    parser.add_argument("disp", help="displacement field in vtk format")
    parser.add_argument("original", help="original image used for registration")

    args = parser.parse_args()
    T_file = args.tensors
    disp_file = args.disp
    I_file = args.original

    # load disp and original image
    _,disp,_,_ = read_data(disp_file)
    xI,I,_,_ = read_data(I_file)
    dv = [(x[1]-x[0]) for x in xI]

    # get transformed coordinates
    Xin = np.stack(np.meshgrid(xI[0],xI[1],xI[2], indexing='ij'))
    X = disp[0] + Xin
    
    # load tensor field from .nii.gz
    T_ = nib.load(T_file)
    # get tensor data
    T = T_.get_fdata()
    # get tensor data coordinates
    T_head = T_.header
    dim = T_head['dim'][1:4]
    pixdim = T_head['pixdim'][1:4]

    xT = []
    for i in range(len(dim)):
        x = (np.arange(dim[i]) - (dim[i]-1)/2) * pixdim[i]
        xT.append(x)
    
    Ts = []
    for i in range(6):
        # Resample components of the tensor field at transformed points
        x = interp(xT, T[...,i][None], X)
        Ts.append(x)
    # T has the 6 unique elements of the symmetric positive-definite diffusion tensors.
    # The lower triangle of the matrix must be filled and last 2 dimensions reformed into 3x3 tensors.
    T = np.stack((Ts[0],Ts[3],Ts[4],Ts[3],Ts[1],Ts[5],Ts[4],Ts[5],Ts[2]), axis=-1)
    T = T.reshape(T.shape[:-1]+(3,3)) # result is row x col x slice x 3x3

    # calculate jacobian of transformed coordinates
    jacobian = lambda X,dv : np.stack(np.gradient(X[0], dv[0],dv[1],dv[2], axis=(1,2,3))).transpose(2,3,4,0,1)
    J = jacobian(X, dv)

    # transform tensor field using preservation of principle directions (PPD) method
    Q_ppd = ppd(T, J)
    T_ppd = Q_ppd @ T @ Q_ppd.transpose(0,1,2,4,3)

    return T_ppd


if __name__== "__main__":
    main()

