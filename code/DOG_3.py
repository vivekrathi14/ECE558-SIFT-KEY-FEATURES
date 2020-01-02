# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 23:00:56 2019

@author: Vivek Rathi
"""

import cv2
import numpy as np
from c_fft import convfft as conv
import matplotlib.pyplot as plt
import math


img_o = cv2.imread("sunflowers.jpg",0) #gray scale conversion
img = np.float32(img_o)


sigma_1 = 2
#sigma_1 = 1.6 # professor slides
k = 1.414

def kernel(sigma):
    kern = int(np.ceil(6*sigma))
    x = cv2.getGaussianKernel(kern,sigma)
    g_k = x*x.T
    return g_k

G_L = []

for i in range(10):
    p = np.power(k,i)
    sigma = sigma_1 * p
    gaus_k = kernel(sigma)
    flt_img = conv(img,gaus_k)
    img = np.copy(flt_img)
#    flt_img = cv2.filter2D(img,-1,gaus_k)
#    diff_img = cv2.subtract(img,flt_img)
    G_L.append(flt_img)

DOG_L = []
for i in range(9):
    diff_img = np.subtract(G_L[i+1],G_L[i])
    DOG_L.append(diff_img)
#
#    
#
#
##for i in range(9):
##    name = "Difference " + str(i)
##    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
##    cv2.imshow(name,DOG_L[i]/DOG_L[i].max())
#
DOG_L_array = np.array([i for i in DOG_L])
co_ordinates = [] #to store co ordinates
r,c = img.shape
for i in range(1,r):
    for j in range(1,c):
        slice_img = DOG_L_array[:,i-1:i+2,j-1:j+2] #9*3*3 slice
        result = np.amax(slice_img) #finding maximum
        if result >= 15:
            z,x,y = np.unravel_index(slice_img.argmax(),slice_img.shape)
            co_ordinates.append((i+x-1,j+y-1,np.round((k**z*sigma_1),2))) #finding co-rdinates

loc_x_y = list(set(co_ordinates))

print(len(loc_x_y))

for i in range(len(loc_x_y)):
    if abs(loc_x_y[i][0] - loc_x_y[i+1][0]) < 5 and abs(loc_x_y[i][1] - loc_x_y[i+1][1]) < 5 :
        a = 1
    

fig, ax = plt.subplots()
nh,nw = img.shape
count = 0

ax.imshow(img_o, interpolation='nearest',cmap="gray")
for blob in co_ordinates:
   y,x,r = blob
   c = plt.Circle((x, y), r*1.414, color='red', linewidth=1, fill=False)
   ax.add_patch(c)
ax.plot()  
plt.show()
#
