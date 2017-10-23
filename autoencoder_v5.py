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
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from scipy.ndimage.interpolation import rotate,shift
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
n = imgs_train.shape[0]-1
x_train = np.ndarray((n,tdimn[0],tdimn[1],3))
x_train_noisy = np.ndarray((n,tdimn[0],tdimn[1],3))

for i in xrange(x_train.shape[0]):
    x_train[i] = imgs_train[-1]
    x_train_noisy[i] = imgs_train[i]

#%%
xr = []
xrn = []
angles = [15.0,30.0,45.0]
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

#%%
test = np.concatenate((x_train[0:2],x_train[7:9],x_train[14:16],x_train[21:23]),axis=0)
    


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
autoencoder.fit(test, test, batch_size=1,
                epochs=200,
                verbose = 1,
                shuffle=True)
#callbacks=[TensorBoard(log_dir='autoencoder', histogram_freq=0, write_graph=True)]
#%%
#x_out = autoencoder.predict(x_train_noisy[0:5])
x_out = autoencoder.predict(test[0:3])
#%%
for i in range(x_out.shape[0]):
    plt.figure(figsize=(12,10))
    plt.imshow(x_out[i])

