#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:59:22 2017

@author: anand
"""

import pandas as pd
import os
from pymasker import LandsatMasker
from pymasker import LandsatConfidence
from scipy import misc
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from U_Net1 import myUnet
from skimage import io
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import params


#%%
dir1 = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(dir1,'doc_parser.csv'))
df = df.astype(str)
df['P'] = df['P'].apply(lambda x:x.zfill(3))
df['BQA'] =str(dir1)+"/"+df['GP']+"/"+df['P']+"/"+df['Child']+"/"+df['Child']+"_BQA.TIF"
df['B4'] = str(dir1)+"/"+df['GP']+"/"+df['P']+"/"+df['Child']+"/"+df['Child']+"_B4.TIF"
df['B3'] = str(dir1)+"/"+df['GP']+"/"+df['P']+"/"+df['Child']+"/"+df['Child']+"_B3.TIF"
df['B2'] = str(dir1)+"/"+df['GP']+"/"+df['P']+"/"+df['Child']+"/"+df['Child']+"_B2.TIF"
df['Dir_Path'] = str(dir1)+"/"+df['GP']+"/"+df['P']+"/"+df['Child']

tdimn = (512,512)
#%%
def img_preprocess(qa,b4,b3,b2, pt):    
    masker = LandsatMasker(qa,collection=1)
    conf = LandsatConfidence.high
    mask1 = masker.get_cirrus_mask(conf)
    mask2 = masker.get_cloud_mask(conf)
    mask = np.bitwise_or(mask1,mask2)
    mask = misc.imresize(mask,tdimn)
    
    
    img1 = gdal.Open(b4)
    orig1 = img1.ReadAsArray()
    orig1 = misc.imresize(orig1,tdimn)
    orig1 = orig1[:,:,np.newaxis]
    
    img2 = gdal.Open(b3)
    orig2 = img2.ReadAsArray()
    orig2 = misc.imresize(orig2,tdimn)
    orig2 = orig2[:,:,np.newaxis]
    
    img3 = gdal.Open(b2)
    orig3 = img3.ReadAsArray()
    orig3 = misc.imresize(orig3,tdimn)
    orig3 = orig3[:,:,np.newaxis]
    
    orig = np.concatenate((orig1,orig2,orig3),axis=2)
# =============================================================================
#     io.imsave(os.path.join(pt,'rgb3.png'),orig)
#     img_t = load_img(os.path.join(pt,'rgb3.png'),grayscale=True)
#     img_t = img_to_array(img_t)
# =============================================================================
    
    plt.figure(figsize=(40,20))
    plt.subplot(131)
    plt.imshow(orig,cmap = plt.get_cmap('gist_gray'))
    plt.imshow(mask, cmap = plt.get_cmap('Reds'), alpha=0.3)
    plt.title('Cloud-2')
    plt.show()     
    
    img_m = np.reshape(mask,(mask.shape[0],mask.shape[1],1))
    #print(img_t.shape,img_m.shape)
    return (orig, img_m)

#%%
def create_training_data(imin,imax):
    imgdatas = np.ndarray(((imax-imin),tdimn[0],tdimn[1],3))
    imgmasks = np.ndarray(((imax-imin),tdimn[0],tdimn[1],1))
    for i in xrange(imgdatas.shape[0]):
        qa,b4,b3,b2,pt = df['BQA'].iloc[i],df['B4'].iloc[i],df['B3'].iloc[i],df['B2'].iloc[i],df['Dir_Path'].iloc[i]
        imgs_train, imgs_mask_train = img_preprocess(qa,b4,b3,b2,pt)
        imgdatas[i] = imgs_train
        imgmasks[i] = imgs_mask_train
    imgdatas = imgdatas.astype('float32')
    imgmasks = imgmasks.astype('float32')
    imgdatas /= 255
    #mean = imgdatas.mean(axis=0)
    #imgdatas -= mean
    imgmasks /= 255
    imgmasks[imgmasks > 0.5] = 1
    imgmasks[imgmasks <= 0.5] = 0
    
    print('completed loading training data')
    return (imgdatas,imgmasks)
#%%
def create_test_data(imin,imax):
    imgdatas = np.ndarray(((imax-imin),tdimn[0],tdimn[1],3))
    #imgmasks = np.ndarray(((imax-imin),tdimn[0],tdimn[1],1))
    for i in xrange(imgdatas.shape[0]):
        qa,b4,b3,b2,pt = df['BQA'].iloc[i],df['B4'].iloc[i],df['B3'].iloc[i],df['B2'].iloc[i],df['Dir_Path'].iloc[i]
        imgs_test, imgs_mask_test = img_preprocess(qa,b4,b3,b2,pt)
        imgdatas[i] = imgs_test
        #imgmasks[i] = imgs_mask_train
    imgdatas = imgdatas.astype('float32')
    imgdatas/= 255
    #mean = imgdatas.mean(axis=0)
    #imgdatas -= mean
    print('completed loading test data')
    return (imgdatas)

#%%
#%%
imgs_test = create_test_data(8,11)
model = params.model_factory()
imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

#%%
model.load_weights('best_weights.hdf5')
imgs_mask_test2 = model.predict(imgs_test, batch_size=1, verbose=1)

#%%
imgs_train,imgs_mask_train = create_training_data(0,6)

