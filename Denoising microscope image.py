# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 01:20:37 2021

@author: abc
"""

from skimage import io,img_as_float
from scipy import ndimage as nd
from matplotlib import pyplot as plt
import numpy as np

img = img_as_float(io.imread("C:\\Users\\abc\\Desktop\\image\\aeroplane\\test_image.jpg"))

gaussian_img = nd.gaussian_filter(img, sigma=3)
plt.imsave("C:\\Users\\abc\\Desktop\\image\\aeroplane\\gaussian.jpg",gaussian_img)

median_img = nd.median_filter(img,size = 3)
plt.imsave("C:\\Users\\abc\\Desktop\\image\\aeroplane\\median.jpg",median_img)




from skimage.restoration import denoise_nl_means, estimate_sigma
sigma_est = np.mean(estimate_sigma(img, multichannel=True))
denoise = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False,
                           patch_size=5,patch_distance=6,multichannel=True)

plt.imsave("C:\\Users\\abc\\Desktop\\image\\aeroplane\\denoise.jpg",denoise)
