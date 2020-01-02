# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 00:12:45 2019
sunflowers - 249
fishes - 235
einstein - 240
butterfly - 240
cookies - 245
mm - 240
buttons - 240
balloons - 240

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from c_fft import convfft as conv
from pylab import *
from scipy import ndimage
from scipy.ndimage import filters
from scipy import spatial


def kernel_log(sigma):
    #window size 
#    n = int(4*(sigma+0.5))
    n = np.ceil(sigma*6)
    y,x = np.ogrid[-n//2:n//2+1,-n//2:n//2+1]
    y_kernel = np.exp(-(y*y/(2.*sigma*sigma)))
    x_kernel = np.exp(-(x*x/(2.*sigma*sigma)))
    kernel = (-(2*sigma**2) + (x*x + y*y) ) *  (x_kernel*y_kernel) * (1/(2*np.pi*sigma**4))
    return kernel


def scalespace(img):
    scale_imgs = [] #to store responses
    for i in range(9):
        sigma_r = sigma*np.power(k,i) #sigma 
        scale = kernel_log(sigma_r) #kernel generation
#        image = cv2.filter2D(img,-1,scale) # convolving image
        image = conv(img,scale)
        image = np.square(image) # squaring the response
        scale_imgs.append(image)
    scale_imgs_mat = np.array([i for i in scale_imgs]) # storing the #in numpy array
    return scale_imgs_mat


def lap_blob(scale_img):
    loc = [] 
    r,c = img.shape
    for i in range(1,r):
        for j in range(1,c):
            img_nd = scale_img[:,i-1:i+2,j-1:j+2] #9*3*3 slice
            peak = np.max(img_nd) #finding maximum
            if peak >= 245: #229 -threshold
                z,x,y = np.unravel_index(img_nd.argmax(),img_nd.shape)
                loc.append((i+x-1,j+y-1,(k**z)*sigma)) #finding co-rdinates
    return loc

def blob_overlap(blob1, blob2):
    n_dim = len(blob1) - 1
    root_ndim = sqrt(n_dim)
    #print(n_dim)
    
    # radius of two blobs
    r1 = blob1[-1] * root_ndim
    r2 = blob2[-1] * root_ndim
    
    d = sqrt(np.sum((blob1[:-1] - blob2[:-1])**2))
    
    #no overlap between two blobs
    if d > r1 + r2:
        return 0
    # one blob is inside the other, the smaller blob must die
    elif d <= abs(r1 - r2):
        return 1
    else:
        #computing the area of overlap between blobs
        ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
        ratio1 = np.clip(ratio1, -1, 1)
        acos1 = math.acos(ratio1)
        ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
        ratio2 = np.clip(ratio2, -1, 1)
        acos2 = math.acos(ratio2)
        a = -d + r2 + r1
        b = d - r2 + r1
        c = d + r2 - r1
        d = d + r2 + r1
        area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 -0.5 * sqrt(abs(a * b * c * d)))
        return area/(math.pi * (min(r1, r2) ** 2))
    
def NMS(blobs_array, overlap):
    sigma = blobs_array[:, -1].max()
    distance = 2 * sigma * sqrt(blobs_array.shape[1] - 1)
    tree = spatial.cKDTree(blobs_array[:, :-1])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            if blob_overlap(blob1, blob2) > overlap:
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0
    return np.array([b for b in blobs_array if b[-1] > 0])

#start time
s1 = time.time()

k = 1.414
sigma = 1

#read image
img = cv2.imread("einstein.jpg",0) #gray scale conversion


scale_images_np = scalespace(img)

loc_x_y = list(set(lap_blob(scale_images_np)))


print(len(loc_x_y))


loc_f = np.array(loc_x_y)
loc_f = NMS(loc_f,0.5)


fig, ax = plt.subplots()
nh,nw = img.shape


ax.imshow(img, interpolation='nearest',cmap="gray")
for blob in loc_f:
    y,x,r = blob
    c = plt.Circle((x, y), r*1.414, color='red', linewidth=1.5, fill=False)
    ax.add_patch(c)
ax.plot()  
plt.show()
