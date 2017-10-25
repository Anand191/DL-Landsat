#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:21:04 2017

@author: anand
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 15:25:38 2017

@author: anand
"""

import numpy as np
import pandas as pd
from keras.models import *
from keras.layers import Input,Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, BatchNormalization
from keras.layers import merge
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.layers.merge import concatenate
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from pymasker import LandsatMasker
from pymasker import LandsatConfidence
from scipy import misc
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
count = 0
for filename in os.listdir('All_Imgs'):
    if filename.endswith(".png"):
        count += 1
idx = np.arange(count)
lim = int(0.75*len(idx))
idx1 = idx[0:lim]
idx2 = idx[lim:]
np.random.shuffle(idx1)


#%%
def img_preprocess(i,qa):    
    masker = LandsatMasker(qa,collection=1)
    conf = LandsatConfidence.high
    mask1 = masker.get_cirrus_mask(conf)
    mask2 = masker.get_cloud_mask(conf)
    mask = np.bitwise_or(mask1,mask2)
    mask = misc.imresize(mask,tdimn)
    
    orig = misc.imread(os.path.join(dir1,"All_Imgs/orig_{}.png".format(i)))

# =============================================================================
#     plt.figure(figsize=(40,20))
#     plt.subplot(131)
#     plt.imshow(orig)
#     plt.imshow(mask, cmap = plt.get_cmap('Reds'), alpha=0.3)
#     plt.title('Cloud-2')
#     plt.show()     
# =============================================================================
    img_m = np.reshape(mask,(mask.shape[0],mask.shape[1],1))
    return (orig, img_m)

  
def create_training_data(idx1):
    imgdatas = np.ndarray((len(idx1),tdimn[0],tdimn[1],3))
    imgmasks = np.ndarray((len(idx1),tdimn[0],tdimn[1],1))
    j = 0
    for i in idx1:
        qa = df['BQA'].iloc[i]
        imgs_train, imgs_mask_train = img_preprocess(i,qa)
        imgdatas[j] = imgs_train
        imgmasks[j] = imgs_mask_train
        j += 1
    imgdatas = imgdatas.astype('float32')
    imgmasks = imgmasks.astype('float32')
    imgdatas /= 255.

    imgmasks /= 255.
    imgmasks[imgmasks > 0.5] = 1
    imgmasks[imgmasks <= 0.5] = 0
    
    for j in xrange(imgdatas.shape[0]):
        plt.figure(figsize=(40,20))
        plt.subplot(131)
        plt.imshow(imgdatas[j])
        plt.imshow(np.squeeze(imgmasks[j],axis=2), cmap = plt.get_cmap('Reds'), alpha=0.3)
        plt.title('Cloud-2')
        plt.show() 
    
    print('completed loading training data')
    return (imgdatas,imgmasks)


def create_test_data(idx2):
    imgdatas = np.ndarray((len(idx2),tdimn[0],tdimn[1],3))
    #imgmasks = np.ndarray(((imax-imin),tdimn[0],tdimn[1],1))
    j = 0
    for i in idx2:
        qa = df['BQA'].iloc[i]
        imgs_test, imgs_mask_test = img_preprocess(i,qa)
        imgdatas[j] = imgs_test
        #imgmasks[i] = imgs_mask_train
        j += 1
    imgdatas = imgdatas.astype('float32')
    imgdatas/= 255.
    for j in xrange(imgdatas.shape[0]):
        plt.figure(figsize=(12,10))
        plt.imshow(imgdatas[j])
        plt.show()
    print('completed loading test data')
    return (imgdatas)    

#%%
class myUnet(object):
    def __init__(self, img_rows = 512, img_cols = 512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self,idx1,idx2):
        #mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = create_training_data(idx1)
        imgs_test = create_test_data(idx2)
        return imgs_train, imgs_mask_train, imgs_test

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols,3))
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
        
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6],axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        
        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7],axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        
        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8],axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        
        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9],axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
        
        model = Model(input = inputs, output = conv10)
        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        print (model.summary())
        
        return model

    def train(self,idx1,idx2):
        print("loading data")
        imgs_train,imgs_mask_train,imgs_test = self.load_data(idx1,idx2)
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        model.load_weights('unet_v3.h5')
        
        model_checkpoint = ModelCheckpoint('unet_v2.h5', monitor='loss',verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=20, verbose=1, 
          shuffle=True, callbacks=[model_checkpoint])
        
        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        for j in xrange(imgs_test.shape[0]):                
            plt.figure(figsize=(14,10))
            plt.subplot(211)
            plt.imshow(imgs_test[j],cmap=plt.get_cmap('gist_gray'))
            plt.subplot(212)
            plt.imshow(np.squeeze(imgs_mask_test[j],axis=2),cmap=plt.get_cmap('Greys'))
            
        imgs_mask_test[imgs_mask_test > 0.5] = 1
        imgs_mask_test[imgs_mask_test <= 0.5] = 0
        for j in xrange(imgs_test.shape[0]):                
            plt.figure(figsize=(14,10))
            plt.subplot(111)
            plt.imshow(np.squeeze(imgs_mask_test[j],axis=2),cmap=plt.get_cmap('Greys'))
            
#%%
if __name__ == '__main__':    
    myunet = myUnet()
    myunet.train(idx1,idx2)
