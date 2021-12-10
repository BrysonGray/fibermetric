# %%
from scipy.ndimage import gaussian_filter, sobel
from skimage import color
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# %%
def rgb2gray(rgb):
    return np.dot(rgb[...,:], [0.2989, 0.5870, 0.1140])

img = 'pygmy_goats.jpg'

I = io.imread(img)
if len(I.shape) == 3:
    I = rgb2gray(I)
# fit I to range (0,1)
I = I * 1/255.
plt.figure()
plt.imshow(I, cmap='gray')
plt.title('Original')

# sobel function gets approx. gradient of image intensity along the specified axis
I_y = sobel(I, axis=0)
I_x = sobel(I, axis=1)

# construct the structure tensor, S
I_y_sq = I_y**2
I_x_sq = I_x**2
I_xy = I_x*I_y
S = np.stack((gaussian_filter(I_x_sq, 3), gaussian_filter(I_xy, 3),
    gaussian_filter(I_xy, 3), gaussian_filter(I_y_sq, 3)), axis=-1)

# construct anisotropy index, AI
d,v = np.linalg.eig(S.reshape((I.shape[0],I.shape[1],2,2)))

# get only principle eigenvectors
max_idx = np.abs(d).argmax(axis=-1)
max_idx = np.ravel(max_idx)
max_idx = np.array([np.arange(max_idx.shape[0]), max_idx])
v = np.transpose(v, axes=(0,1,3,2)).reshape(-1,2,2)
v = v[max_idx[0],max_idx[1]].reshape(I.shape[0],-1,2)

theta = ((np.arctan(v[...,1] / v[...,0])) + np.pi / 2) / np.pi
# %%
AI = abs(d[...,0] - d[...,1]) / (d[...,0] + d[...,1])

# get primary orientation, theta
# TODO: this may not be quite what we want (color cycles every pi/2 rather than every pi)
# theta = 0.5 * np.arctan(2 * S[...,1] / (S[...,3] - S[...,0]))
# theta = theta * 2 / np.pi + 0.5

# make hsv image where hue= primary orientation, saturation= anisotropy, value= original image
stack = np.stack([theta,AI,I],axis=-1)
hsv = matplotlib.colors.hsv_to_rgb(stack)


plt.figure()
plt.imshow(theta, cmap='hsv')
plt.title('theta')

plt.figure()
plt.imshow(AI, cmap='gray')
plt.title('Anisotropy Index')

plt.figure()
plt.imshow(hsv)
plt.title('hsv')

plt.show()
# %%
