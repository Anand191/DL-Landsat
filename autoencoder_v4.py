#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 21:16:17 2017

@author: anand
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pymasker import LandsatMasker
from pymasker import LandsatConfidence
from scipy import misc
from osgeo import gdal
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage import io
#%%

dir1 = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(dir1,'doc_parser2.csv'))
df = df.astype(str)
df['P'] = df['P'].apply(lambda x:x.zfill(3))
df['BQA'] =str(dir1)+"/"+df['P']+"/"+df['Child']+"/"+df['Child']+"_BQA.TIF"
df['B4'] = str(dir1)+"/"+df['P']+"/"+df['Child']+"/"+df['Child']+"_B4.TIF"
df['B3'] = str(dir1)+"/"+df['P']+"/"+df['Child']+"/"+df['Child']+"_B3.TIF"
df['B2'] = str(dir1)+"/"+df['P']+"/"+df['Child']+"/"+df['Child']+"_B2.TIF"
df['Dir_Path'] = str(dir1)+"/"+df['P']+"/"+df['Child']

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
#     io.imsave(os.path.join(pt,'rgb2.png'),orig)
#     img_t = load_img(os.path.join(pt,'rgb2.png'),grayscale=True)
#     img_t = img_to_array(img_t)
# =============================================================================
    
# =============================================================================
#     plt.figure(figsize=(40,20))
#     plt.subplot(131)
#     plt.imshow(orig,cmap = plt.get_cmap('gist_gray'))
#     plt.imshow(mask, cmap = plt.get_cmap('Reds'), alpha=0.3)
#     plt.title('Cloud-2')
#     plt.show()     
# =============================================================================
    
    img_m = np.reshape(mask,(mask.shape[0],mask.shape[1],1))
    #print(img_t.shape,img_m.shape)
    return (orig, img_m)

#%%
def create_training_data(ir):
    imgdatas = np.ndarray((len(ir),tdimn[0],tdimn[1],3))
    imgmasks = np.ndarray((len(ir),tdimn[0],tdimn[1],1))
    j = 0
    for i in ir:
        qa,b4,b3,b2,pt = df['BQA'].iloc[i],df['B4'].iloc[i],df['B3'].iloc[i],df['B2'].iloc[i],df['Dir_Path'].iloc[i]
        imgs_train, imgs_mask_train = img_preprocess(qa,b4,b3,b2,pt)
        imgdatas[j] = imgs_train
        imgmasks[j] = imgs_mask_train
        j += 1
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


imgs_train, imgs_mask_train = create_training_data([0,1,2,3,6,8,10,4])


#%%
# =============================================================================
# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
# =============================================================================
datagen = ImageDataGenerator(
        rotation_range=10,
        rescale=1./255,
        fill_mode='nearest')

def rotate(i):
    j = 0
    for batch in datagen.flow(imgs_train[i].reshape((1,) + imgs_train[i].shape), batch_size=1, save_to_dir='preview',
                              save_prefix='cloud_{}'.format(i),save_format='jpeg',seed=42):
         j += 1
         if (j >= 2):
             break
for i in range(imgs_train.shape[0]-1):
    rotate(i)

k = 0
for tbatch in datagen.flow(imgs_train[-1].reshape((1,) + imgs_train[-1].shape), batch_size=1, save_to_dir='preview2',
                              save_prefix='tcloud',save_format='jpeg',seed=42):
    k += 1
    if (k>=2):
        break
    
#%%
fname1 = []
for filename in os.listdir('preview'):
    if filename.endswith(".jpeg"):
        fname1.append(filename)
fname1 = np.sort(fname1)

fname2 = []
for filename in os.listdir('preview2'):
    if filename.endswith(".jpeg"):
        fname2.append(filename)
fname2 = np.sort(fname2)


#%%
x_train_noisy = []
for f in fname1:
    img = load_img('preview/{}'.format(f))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x_train_noisy.append(x)
x_train_noisy = np.asarray(x_train_noisy)
x_train_noisy = np.squeeze(x_train_noisy,axis=1)

#%%
x_train = []
for f in fname2:
    img = load_img('preview2/{}'.format(f))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x_train.append(x)
x_train = np.asarray(x_train)
x_train = np.squeeze(x_train,axis=1)
x_train = np.tile(x_train,(7,1,1,1))


#%%
for i in range(x_train.shape[0]-2):
    if(np.array_equal(x_train[i],x_train[i+1])):
        print ("wrong")
    elif(np.array_equal(x_train[i],x_train[i+2])):
        print ("correct")
         

#%% Autoencoder
input_img = Input(shape=(512,512, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(256,(3,3),activation='relu',padding='same')(x)
x = MaxPooling2D((2,2),padding='same')(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)   

# at this point the representation is (7, 7, 32)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(128,(3,3), activation='relu',padding='same')(x)
x = UpSampling2D((2,2))(x)

x = Conv2D(64,(3,3), activation='relu',padding='same')(x)
x = UpSampling2D((2,2))(x)

decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print autoencoder.summary()

#%%
autoencoder.fit(x_train_noisy, x_train, batch_size=1,
                epochs=10,
                verbose = 1,
                shuffle=True)
#callbacks=[TensorBoard(log_dir='autoencoder', histogram_freq=0, write_graph=True)]
#%%
x_out = autoencoder.predict(x_train_noisy[0:3])
#%%
for i in range(x_out.shape[0]):
    plt.figure(figsize=(12,10))
    plt.imshow(x_out[i])

