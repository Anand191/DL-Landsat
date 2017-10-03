#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 13:39:03 2017

@author: anand
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from osgeo import gdal
from pymasker import LandsatMasker
from pymasker import LandsatConfidence
from skimage import io
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.utils import to_categorical

#%%
masker = LandsatMasker("LC08_L1TP_210017_20170711_20170711_01_RT_BQA.TIF",collection=1)
conf = LandsatConfidence.high
mask1 = masker.get_cirrus_mask(conf)
mask2 = masker.get_cloud_mask(conf)
mask = np.bitwise_or(mask1,mask2)
mask = misc.imresize(mask,(64,64))
plt.imshow(mask,cmap=plt.get_cmap('Reds'))

#%%
img1 = gdal.Open("LC08_L1TP_210017_20170711_20170711_01_RT_B4.TIF")
orig1 = img1.ReadAsArray()
orig1 = misc.imresize(orig1,(64,64))
orig1 = orig1[:,:,np.newaxis]

img2 = gdal.Open("LC08_L1TP_210017_20170711_20170711_01_RT_B3.TIF")
orig2 = img2.ReadAsArray()
orig2 = misc.imresize(orig2,(64,64))
orig2 = orig2[:,:,np.newaxis]

img3 = gdal.Open("LC08_L1TP_210017_20170711_20170711_01_RT_B2.TIF")
orig3 = img3.ReadAsArray()
orig3 = misc.imresize(orig3,(64,64))
orig3 = orig3[:,:,np.newaxis]

orig = np.concatenate((orig1,orig2,orig3),axis=2)
io.imsave('rgb2.png',orig)
img = io.imread('rgb2.png')


#%%
plt.figure(figsize=(40,20))
plt.subplot(131)
plt.imshow(img)
plt.title('Cloud-1')
plt.show()

#%%
plt.figure(figsize=(40,20))
plt.subplot(131)
plt.imshow(np.squeeze(orig1,axis=2),cmap = plt.get_cmap('gist_gray'))
plt.imshow(mask, cmap = plt.get_cmap('Reds'), alpha=0.5)
plt.title('Cloud-2')
plt.show()

#%%patch extraction
img = np.reshape(img,(1,64,64,3))
img2 = np.reshape(mask,(1,64,64,1))
with tf.Session() as sess:
     patches =  tf.extract_image_patches(images=img, ksizes=[1, 8, 8, 1], strides=[1, 1, 1, 1], 
                                         rates=[1, 1, 1, 1], padding='SAME')
     patches = tf.reshape(patches,[-1,8,8,3])
     
     patches2 =  tf.extract_image_patches(images=img2, ksizes=[1, 8, 8, 1], strides=[1, 1, 1, 1], 
                                         rates=[1, 1, 1, 1], padding='SAME')
     patches2 = tf.reshape(patches2,[-1,8,8,1])
     
     val = sess.run(patches)
     val2 = sess.run(patches2)
     
     
     
#%%data preparation
cloud = []
bk = []
for i in xrange(val2.shape[0]):
    if(val2[i,val2.shape[1]/2,val2.shape[2]/2,:][0] != 0):
        cloud.append(val[i,:,:,:].flatten())
    else:
        bk.append(val[i,:,:,:].flatten())

rel = np.column_stack((cloud,np.ones(len(cloud),dtype=int)))
irrel = np.column_stack((bk,np.zeros(len(bk),dtype=int)))
fdata = np.concatenate((rel,irrel),axis=0)
np.random.shuffle(fdata)
#%%
patch_height, patch_width = 8,8
epochs = 10
batch_size =  8

num_classes = 2

input_shape = (patch_width,patch_height,3)
x_train = fdata[:,0:192].reshape(fdata.shape[0], patch_width, patch_height, 3)
y_train = to_categorical(fdata[:,-1],num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Dense(num_classes, activation='softmax'))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy',
              optimizer= adam,
              metrics=['accuracy'])
print(model.summary())
#%%
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_train, y_train))

score = model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#%%
with tf.Session() as sess:
    patches3 =  tf.extract_image_patches(images=img, ksizes=[1, 8,8, 1], strides=[1, 1, 1, 1], 
                                         rates=[1, 1, 1, 1], padding='SAME')
    patches3 = tf.reshape(patches3,[-1,8,8,3])
    val3 = sess.run(patches3)
#%%
dims = mask.shape
d = {0:0,1:1}
mask_n = np.zeros(dims)
z = 0
y = 0
batch = mask.shape[0]
while(y<mask.shape[0]):
    pred = model.predict(val3[z:z+batch,:,:,:])
    for i in xrange(mask.shape[0]):
        mask_n[i,y] = d[np.argmax(pred[i])]
    z += batch
    y += 1
    if (y % 8 == 0):
        print("No. of iterations till results = {}".format(1024-y))
    
#%%
plt.figure(figsize=(40,20))
plt.subplot(131)
plt.imshow(np.squeeze(orig1,axis=2),cmap = plt.get_cmap('gist_gray'))
plt.imshow(mask_n, cmap = plt.get_cmap('Reds'), alpha=0.5)
plt.title('Cloud-3')
plt.show()



