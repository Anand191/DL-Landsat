#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:32:44 2017

@author: anand
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:57:32 2017

@author: anand
"""

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
from keras.layers import Input,Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from scipy.ndimage.interpolation import rotate,shift
#%%

dir1 = os.path.dirname(__file__)
tdimn = (512,512)

#%%
def img_preprocess(i):    
    orig = misc.imread(os.path.join(dir1,"Autoencoder_Imgs/orig_{}.png".format(i)))
    plt.imshow(orig)
    return (orig)

#%%
def create_training_data(ir):
    imgdatas = np.ndarray((len(ir),tdimn[0],tdimn[1],3))
    j = 0
    for i in ir:
        imgs_train = img_preprocess(i)
        imgdatas[j] = imgs_train
        j += 1
    imgdatas = imgdatas.astype('float32')
    imgdatas /= 255
    print('completed loading training data')
    return (imgdatas)


imgs_train1 = create_training_data([0,1,2,3,4,5,6,7,8,22,29,31])
imgs_train2 = create_training_data([9,10,11,12,13,14,15,32])
imgs_train3 = create_training_data([16,17,18,19,20,21,23,24,25,26,27,28,33])
master = [imgs_train1,imgs_train2,imgs_train3]


#%%
def create_data(imgs_train):    
    n = imgs_train.shape[0]-1
    x_train = np.ndarray((n,tdimn[0],tdimn[1],3))
    x_train_noisy = np.ndarray((n,tdimn[0],tdimn[1],3))
    
    for i in xrange(x_train.shape[0]):
        x_train[i] = imgs_train[-1]
        x_train_noisy[i] = imgs_train[i]
    return(x_train,x_train_noisy)

#%%
def rot(x_train,x_train_noisy):
    xr = []
    xrn = []
    angles = [45.0]
    for angle in angles:
        xt = rotate(x_train,angle=angle,axes=(1,2),reshape=False)
        xtn = rotate(x_train_noisy,angle=angle,axes=(1,2),reshape=False)
        xr.append(xt)
        xrn.append(xtn)
    
    xr = np.asarray(xr)
    xrn = np.asarray(xrn)
    
    xr = xr.reshape((xr.shape[0]*xr.shape[1],tdimn[0],tdimn[1],3))
    xrn = xrn.reshape((xrn.shape[0]*xrn.shape[1],tdimn[0],tdimn[1],3))
    
    
    x_train = np.concatenate((x_train,xr),axis=0)
    x_train_noisy = np.concatenate((x_train_noisy,xrn),axis=0)
    
    return (x_train,x_train_noisy)

#%%

#test = np.concatenate((x_train[0:2],x_train[7:9],x_train[14:16],x_train[21:23]),axis=0)
X_train = []
X_train_noisy = []
for slave in master:
    x_train,x_train_noisy = create_data(slave)
    x_t, x_t_n = rot(x_train,x_train_noisy)
    X_train.append(x_t)
    X_train_noisy.append(x_t_n)
#%%
X_train = np.concatenate((X_train[0],X_train[1],X_train[2]),axis=0)
X_train_noisy = np.concatenate((X_train_noisy[0],X_train_noisy[1],X_train_noisy[2]),axis=0)

#%% Autoencoder
input_img = Input(shape=(512,512, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(256,(3,3),activation='relu',padding='same', kernel_initializer = 'he_normal')(x)
x = MaxPooling2D((2,2),padding='same')(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)   

# at this point the representation is (7, 7, 32)
x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(encoded)
x = UpSampling2D((2, 2))(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(128,(3,3), activation='relu',padding='same', kernel_initializer = 'he_normal')(x)
x = UpSampling2D((2,2))(x)

x = Conv2D(64,(3,3), activation='relu',padding='same', kernel_initializer = 'he_normal')(x)
x = UpSampling2D((2,2))(x)

decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print autoencoder.summary()

#%%
autoencoder.load_weights('autoencoderv7.h5')
model_checkpoint = ModelCheckpoint('autoencoderv8.h5', monitor='loss',verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='autoencoder',write_graph=True)
autoencoder.fit(X_train_noisy, X_train, batch_size=2,
                epochs=15,
                verbose = 1,
                shuffle=True,
                callbacks = [model_checkpoint,tb])
#%%
#x_out = autoencoder.predict(x_train_noisy[0:5])
idx = np.arange(X_train.shape[0])
idx1 = np.random.choice(idx,5)
x_out = autoencoder.predict(X_train_noisy[idx1])
#%%
for j in idx1:
    plt.figure(figsize=(12,10))
    plt.imshow(X_train_noisy[j])
    
for i in range(x_out.shape[0]):
    plt.figure(figsize=(12,10))
    plt.imshow(x_out[i])
    

