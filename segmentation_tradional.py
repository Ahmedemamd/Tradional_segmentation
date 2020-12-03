# -*- coding: utf-
#Created on Sun Nov 29 14:58:22 2020

#@author: Ahmed Emam
#Denoising
"""

Works well for random gaussian noise but not as good for salt and pepper
https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/Buades-NonLocal.pdf
The non-local means algorithm replaces the value of a pixel by an average 
of a selection of other pixels values: small patches centered on the other 
pixels are compared to the patch centered on the pixel of interest, and the 
average is performed only for pixels that have patches close to the current patch. 
"""
#!pip install opencv-python
import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt

img_gaussian_noise = img_as_float(io.imread(r'C:\Users\49174\OneDrive\Desktop\1st_pic.jpg', as_gray=True))


img = img_gaussian_noise

sigma_est = np.mean(estimate_sigma(img, multichannel=True))
#sigma_est = 0.1

denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True,
                               patch_size=5, patch_distance=3, multichannel=False)

"""
When the fast_mode argument is False, a spatial Gaussian weighting is applied 
to the patches when computing patch distances. When fast_mode is True a 
faster algorithm employing uniform spatial weighting on the patches is applied.
Larger h allows more smoothing between disimilar patches.
"""





cv2.imshow("Original", img)
cv2.imshow("NLM Filtered", denoise_img)
cv2.waitKey(0)          
cv2.destroyAllWindows() 


###########################################################################################################
#Image deconvolution
#Uses deconvolution to sharpen images. 

import matplotlib.pyplot as plt
from skimage import io, color, restoration, img_as_float

img_orig = img_as_float(io.imread(r'C:\Users\49174\OneDrive\Desktop\1st_pic.jpg', as_gray=True))


#PSF
import scipy.stats as st
import numpy as np

#psf = np.ones((3, 3)) / 9  #point spread function to be used for deconvolution.

#The following page was used as reference to generate the kernel
#https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm

def gkern(kernlen=21, nsig=2):    #Returns a 2D Gaussian kernel.

    lim = kernlen//2 + (kernlen % 2)/2
    x = np.linspace(-lim, lim, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

psf = gkern(5,3)   #Kernel length and sigma
print(psf)
deconvolved, _ = restoration.unsupervised_wiener(img_orig, psf)
##########################################################################################################
#Edge detection using canny edge detector
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import roberts, sobel, scharr, prewitt

img = io.imread(r'C:\Users\49174\OneDrive\Desktop\1st_pic.jpg', as_gray=True)  #Convert to grey scale
#print(img.shape)
#plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')

edge_roberts = roberts(img)
#plt.imshow(edge_roberts, cmap=plt.cm.gray, interpolation='nearest')
edge_sobel = sobel(img)
edge_scharr = scharr(img)
edge_prewitt = prewitt(img)


fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                         figsize=(8, 8))
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(edge_roberts, cmap=plt.cm.gray)
ax[1].set_title('Roberts Edge Detection')

ax[2].imshow(edge_sobel, cmap=plt.cm.gray)
ax[2].set_title('Sobel')

ax[3].imshow(edge_scharr, cmap=plt.cm.gray)
ax[3].set_title('Scharr')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

#Another edge filter is Canny. This is not just a single operation
#It does noise reduction, gradient calculation, and edge tracking among other things. 
#Canny creates a binary file, true or false pixels. 
from skimage import feature
edge_canny = feature.canny(entropy_img, sigma=3)
plt.imshow(edge_canny)


from skimage import feature
edge_canny = feature.canny(img, sigma=3)
plt.imshow(edge_canny)




############################################################################################################
#Entropy
#Entropy quantifies disorder.
#Since cell region has high variation in pixel values the entropy would be
#higher compared to scratch region
from skimage.filters.rank import entropy
from skimage.morphology import disk
entropy_img = entropy(img, disk(5))
plt.imshow(entropy_img,cmap='gray')

#Scratch Analysis - single image
#Now let us use otsu to threshold high vs low entropy regions.
plt.hist(entropy_img.flat, bins=100, range=(0,5))  #.flat returns the flattened numpy array (1D)
from skimage.filters import threshold_otsu

thresh = threshold_otsu(entropy_img)

#Now let us binarize the entropy image 
binary = entropy_img <= thresh
plt.imshow(binary)
#######
from skimage.filters import try_all_threshold



# Here, we specify a radius for local thresholding algorithms.
# If it is not specified, only global algorithms are called.
fig, ax = try_all_threshold(entropy_img, figsize=(10, 8), verbose=False)
plt.show()







################################################################################################################
###########################################################################################################
##GMM Segmentation

img = cv2.imread(r'C:\Users\49174\OneDrive\Desktop\1st_pic.jpg')
img2 = img.reshape((-1,3))
from sklearn.mixture import GaussianMixture as GMM
n = 2
gmm_model = GMM(n, covariance_type='tied').fit(img2)
#The above line generates GMM model for n=2
#Now let us call the bic method (or aic if you want).
bic_value = gmm_model.bic(img2)  #Remember to call the same model name from above)
print(bic_value)  #You should see bic for GMM model generated using n=2.
#Do this exercise for different n values and plot them to find the minimum.
#Now, to explain m.bic, here are the lines I used in the video. 
n_components = np.arange(1,10)
gmm_models = [GMM(n, covariance_type='tied').fit(img2) for n in n_components]
plt.plot(n_components, [m.bic(img2) for m in gmm_models], label='BIC')

##Here, we are computing multiple GMM models each by changing n value from 1 to 10. 
#Then, for each n value we are computing bic via m.bic(img2) where m is replaced by gmm_models for each of the model generated. Think of this as typing gmm_model.bic(img2) each time you change n and generate a new GMM model. 
##I hope this explanation helps better understand the video content. 


import numpy as np
import cv2


#Use plant cells to demo the GMM on 2 components
#Use BSE_Image to demo it on 4 components
#USe alloy.jpg to demonstrate bic and how 2 is optimal for alloy

img = cv2.imread(r'C:\Users\49174\OneDrive\Desktop\1st_pic.jpg')

# Convert MxNx3 image into Kx3 where K=MxN
img2 = img.reshape((-1,3))  #-1 reshape means, in this case MxN

from sklearn.mixture import GaussianMixture as GMM

#covariance choices, full, tied, diag, spherical
gmm_model = GMM(n_components=3, covariance_type='tied').fit(img2)  #tied works better than full
gmm_labels = gmm_model.predict(img2)

#Put numbers back to original shape so we can reconstruct segmented image
original_shape = img.shape
segmented = gmm_labels.reshape(original_shape[0], original_shape[1])
cv2.imwrite(r'C:\Users\49174\OneDrive\Desktop\1st_pic2.jpg', segmented)