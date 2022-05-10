import numpy as np
import nibabel as nib
import argparse
from emlddmm import read_data, interp


# preservation of principle directions
def ppd(tensors,J):
    # define function to construct rotation matrix from axis of rotation and angle
    rot = lambda n, theta : np.array([[np.cos(theta)+n[0]**2*(1-np.cos(theta)), n[0]*n[1]*(1-np.cos(theta))-n[2]*np.sin(theta), n[0]*n[2]*(1-np.cos(theta))+n[1]*np.sin(theta)],
                                    [n[0]*n[1]*(1-np.cos(theta))+n[2]*np.sin(theta), np.cos(theta)+n[1]**2*(1-np.cos(theta)), n[1]*n[2]*(1-np.cos(theta))-n[0]*np.sin(theta)],
                                    [n[0]*n[2]*(1-np.cos(theta))-n[1]*np.sin(theta), n[1]*n[2]*(1-np.cos(theta))+n[0]*np.sin(theta), np.cos(theta)+n[2]**2*(1-np.cos(theta))]])
    Q = np.eye(3)
    # compute unit eigenvectors, e, of tensors
    w,e = np.linalg.eig(tensors)
    # compute unit vectors n1 and n2 in the directions of J@e[0] and J@e[1]
    Je1 = J @ e[0]
    n1 = Je1 / np.linalg.norm(Je1)
    Je2 = J @ e[1]
    n2 = Je2 / np.linalg.norm(Je2)
    # compute a rotation matrix, R1, that maps e[0] onto n1
    theta = np.arccos((e[0] @ n1) / (np.linalg.norm(e[0])*np.linalg.norm(n1)))
    r = np.cross(e[0],n1) / (np.linalg.norm(e[0]) * np.linalg.norm(n1) * np.sin(theta))
    R1 = rot(r,theta)
    # compute a secondary rotation, about n1, to map e[1] from its position after the first rotation, R1 @ e[1],
    # to the n1-n2 plane.
    Pn2 = n2 - (n2 @ n1) @ n1
    Pn2 = Pn2 / np.linalg.norm(Pn2)
    R1e1 = R1 @ e[0]
    R1e2 = R1 @ e[1]
    phi = np.arccos(R1e2 @ Pn2 / (np.linalg.norm(R1e2) * np.linalg.norm(Pn2)))
    R2 = rot(R1e1, phi)
    # rotation matrix is composition of the two rotations, R2 and R1.
    Q = R2 @ R1

    return Q

# finite strain
def fs(J):
    Q = J / (J @ np.transpose(J, axes=(0,1,2,4,3)))**(-1/2)

    return Q

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
    QT_ppd = Q_ppd @ T

    # transform tensor field using finite strain (FS) method
    Q_fs = fs(J)
    QT_fs = Q_fs @ T

    return QT_ppd, QT_fs 


if __name__== "__main__":
    main()

