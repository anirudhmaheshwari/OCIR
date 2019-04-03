
# coding: utf-8

# # Experimenting with normalization and preprocessing of words

# In[1]:


# Import stuff
import os
import sys
import random
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

sys.path.append('../src')
from ocr.helpers import implt
from ocr.datahelpers import load_words_data

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (6.0, 5.0)


# In[2]:


raw_images, _  = load_words_data('../data/raw/breta/words/')
implt(random.choice(raw_images), 'gray', t='Original')


# ## Testing normalization

# In[3]:


def normalization1(img):
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = 255 - th
    img = 255 - img
    implt(th, 'gray')
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
    # convert from BGR to LAB color space
    lab = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2LAB) 
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to B
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, th2 = cv2.threshold(img2, 0, 255, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
    implt(th2, 'gray')
    
    
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(th, kernel, iterations = 1)
    dilation = cv2.GaussianBlur(th,(3,3),0)
    res = cv2.bitwise_and(th2, th2, mask=dilation)
    implt(res, 'gray')

# Produce bad results for some images
normalization1(random.choice(raw_images))


# In[4]:


class HysterThresh:    
    def __init__(self, img):
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = 255 - img
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255        
        hist, bins = np.histogram(img.ravel(), 256, [0,256])
        
        self.high = np.argmax(hist) + 65
        self.low = np.argmax(hist) + 45
        self.diff = 255 - self.high
        
        self.img = img
        self.im = np.zeros(img.shape, dtype=img.dtype)
        self.hyster()
        
    def hyster_rec(self, r, c):
        h, w = self.img.shape
        for ri in [r-1, r+1]:
            for ci in [c-1, c+1]:
                if (h > ri >= 0
                    and w > ci >= 0
                    and self.im[ri, ci] == 0
                    and self.high > self.img[ri, ci] >= self.low):                    
                    self.im[ri, ci] = self.img[ri, ci] + self.diff
                    self.hyster_rec(ri, ci)                      
    
    def hyster(self):
        r, c = self.img.shape
        for ri in range(r):
            for ci in range(c):
                if (self.img[ri, ci] >= self.high):
                    self.im[ri, ci] = 255
                    self.img[ri, ci] = 255
                    self.hyster_rec(ri, ci)
                    
        implt(self.im, 'gray', 'Hister Thresh')


def binary_otsu_norm(img):    
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def bilateral_norm(img):
    img = cv2.bilateralFilter(img, 9, 15, 30)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def histogram_norm(img):
    img = bilateral_norm(img)
    add_img = 255 - cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = 255 - img
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255        
    hist, bins = np.histogram(img.ravel(), 256, [0,256])
    
    img = img.astype(np.uint8)

    ret,thresh4 = cv2.threshold(img,np.argmax(hist)+10,255,cv2.THRESH_TOZERO)
    return add_img
    return cv2.add(add_img, thresh4, dtype=cv2.CV_8UC1)


def normalization2(img):    
    implt(255 - img, 'gray', 'Original')   
    implt(255 - bilateral_norm(img), 'gray', 'Bilateral')
    implt(255 - binary_otsu_norm(img), 'gray', 'Binary OTSU')
    implt(histogram_norm(img), 'gray', 'Binary OTSU + (Filter + TO_ZERO)')
    HysterThresh(cv2.bilateralFilter(img, 10, 10, 30))


normalization2(random.choice(raw_images))


# ## Testing augmentation
# Try using: https://github.com/aleju/imgaug

# In[5]:


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    blur_size = int(4*sigma) | 1
    dx = alpha * cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1),
                                  ksize=(blur_size, blur_size),
                                  sigmaX=sigma)
    dy = alpha * cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1),
                                  ksize=(blur_size, blur_size),
                                  sigmaX=sigma)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    image =  map_coordinates(image, indices, order=1, mode='constant').reshape(shape)
    
    implt(image, 'gray')
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0]+square_size, center_square[1]-square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT)

    return image


# In[6]:


im = 255 - bilateral_norm(random.choice(raw_images))
implt(im, 'gray')
im_trans = elastic_transform(im, im.shape[1] * 2, im.shape[1] * 0.2, im.shape[1] * 0.03)
implt(im_trans, 'gray')

