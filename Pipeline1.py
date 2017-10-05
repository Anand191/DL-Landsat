#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:58:18 2017

@author: anand
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 00:11:49 2017

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
import os
import pandas as pd

dir1 = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(dir1,'doc_parser.csv'))
df = df.astype(str)
df['P'] = df['P'].apply(lambda x:x.zfill(3))
df['BQA'] =str(dir1)+"/"+df['GP']+"/"+df['P']+"/"+df['Child']+"/"+df['Child']+"_BQA.TIF"
df['B4'] = str(dir1)+"/"+df['GP']+"/"+df['P']+"/"+df['Child']+"/"+df['Child']+"_B4.TIF"
df['B3'] = str(dir1)+"/"+df['GP']+"/"+df['P']+"/"+df['Child']+"/"+df['Child']+"_B3.TIF"
df['B2'] = str(dir1)+"/"+df['GP']+"/"+df['P']+"/"+df['Child']+"/"+df['Child']+"_B2.TIF"

tdimn = (256,256)
pn = 15
st = 2
#%%
def img_preprocess(qa,b4,b3,b2, tdimn):    
    masker = LandsatMasker(qa,collection=1)
    conf = LandsatConfidence.high
    mask1 = masker.get_cirrus_mask(conf)
    mask2 = masker.get_cloud_mask(conf)
    mask = np.bitwise_or(mask1,mask2)
    mask = misc.imresize(mask,tdimn)
    #mask = misc.imrotate(mask,18,interp='nearest')
    
    
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
    #orig = misc.imrotate(orig,18,interp='nearest')
    #io.imsave(os.path.join(path,'rgb2.png'),orig)
    #img = io.imread(os.path.join(path,'rgb2.png'))
    
# =============================================================================
#     plt.figure(figsize=(40,20))
#     plt.subplot(131)
#     plt.imshow(orig)
#     plt.title('Cloud-1')
#     
#     plt.figure(figsize=(40,20))
#     plt.subplot(131)
#     plt.imshow(np.squeeze(orig1,axis=2),cmap = plt.get_cmap('gist_gray'))
#     plt.imshow(mask, cmap = plt.get_cmap('Reds'), alpha=0.5)
#     plt.title('Cloud-2')     
# =============================================================================
    
    orig = np.reshape(orig,(1,mask.shape[0],mask.shape[1],3))
    img_m = np.reshape(mask,(1,mask.shape[0],mask.shape[1],1))
    return (orig, img_m)



#%%patch extraction
def train_data():
    cloud_t = []
    bk_t = []
    for i in range(6): 
        qa,b4,b3,b2 = df['BQA'].iloc[i],df['B4'].iloc[i],df['B3'].iloc[i],df['B2'].iloc[i]
        img, img2 = img_preprocess(qa,b4,b3,b2,tdimn)[0], img_preprocess(qa,b4,b3,b2,tdimn)[1]
        with tf.Session() as sess:
             patches =  tf.extract_image_patches(images=img, ksizes=[1, pn, pn, 1], strides=[1, st, st, 1], 
                                                 rates=[1, 1, 1, 1], padding='SAME')
             patches = tf.reshape(patches,[-1,pn,pn,3])
             
             patches2 =  tf.extract_image_patches(images=img2, ksizes=[1, pn, pn, 1], strides=[1, st, st, 1], 
                                                 rates=[1, 1, 1, 1], padding='SAME')
             patches2 = tf.reshape(patches2,[-1,pn,pn,1])
             
             val = sess.run(patches)
             val2 = sess.run(patches2)
             
        for i in xrange(val2.shape[0]):
            if(val2[i,(val2.shape[1]-1)/2,(val2.shape[2]-1)/2,:][0] != 0):
                cloud_t.append(val[i,:,:,:].flatten())
            else:
                bk_t.append(val[i,:,:,:].flatten())

    rel = np.column_stack((cloud_t,np.ones(len(cloud_t),dtype=int)))
    irrel = np.column_stack((bk_t,np.zeros(len(bk_t),dtype=int)))
    fdata = np.concatenate((rel,irrel),axis=0)
    np.random.shuffle(fdata)
    return(fdata)
    
def val_data():
    cloud_t = []
    bk_t = []
    for i in range(6,9): 
        qa,b4,b3,b2 = df['BQA'].iloc[i],df['B4'].iloc[i],df['B3'].iloc[i],df['B2'].iloc[i]
        img, img2 = img_preprocess(qa,b4,b3,b2,tdimn)[0], img_preprocess(qa,b4,b3,b2,tdimn)[1]
        with tf.Session() as sess:
             patches =  tf.extract_image_patches(images=img, ksizes=[1, pn, pn, 1], strides=[1, st, st, 1], 
                                                 rates=[1, 1, 1, 1], padding='SAME')
             patches = tf.reshape(patches,[-1,pn,pn,3])
             
             patches2 =  tf.extract_image_patches(images=img2, ksizes=[1, pn, pn, 1], strides=[1, st, st, 1], 
                                                 rates=[1, 1, 1, 1], padding='SAME')
             patches2 = tf.reshape(patches2,[-1,pn,pn,1])
             
             val = sess.run(patches)
             val2 = sess.run(patches2)
             
        for i in xrange(val2.shape[0]):
            if(val2[i,(val2.shape[1]-1)/2,(val2.shape[2]-1)/2,:][0] != 0):
                cloud_t.append(val[i,:,:,:].flatten())
            else:
                bk_t.append(val[i,:,:,:].flatten())

    rel = np.column_stack((cloud_t,np.ones(len(cloud_t),dtype=int)))
    irrel = np.column_stack((bk_t,np.zeros(len(bk_t),dtype=int)))
    fdata = np.concatenate((rel,irrel),axis=0)
    np.random.shuffle(fdata)
    return(fdata)
    
def test_data():
    cloud_t = []
    bk_t = []
    for i in range(9,11): 
        qa,b4,b3,b2 = df['BQA'].iloc[i],df['B4'].iloc[i],df['B3'].iloc[i],df['B2'].iloc[i]
        img, img2 = img_preprocess(qa,b4,b3,b2,tdimn)[0], img_preprocess(qa,b4,b3,b2,tdimn)[1]
        with tf.Session() as sess:
             patches =  tf.extract_image_patches(images=img, ksizes=[1, pn, pn, 1], strides=[1, 1, 1, 1], 
                                                 rates=[1, 1, 1, 1], padding='SAME')
             patches = tf.reshape(patches,[-1,pn,pn,3])
             
             patches2 =  tf.extract_image_patches(images=img2, ksizes=[1, pn, pn, 1], strides=[1, st, st, 1], 
                                                 rates=[1, 1, 1, 1], padding='SAME')
             patches2 = tf.reshape(patches2,[-1,pn,pn,1])
             
             val = sess.run(patches)
             val2 = sess.run(patches2)
             
        for i in xrange(val2.shape[0]):
            if(val2[i,(val2.shape[1]-1)/2,(val2.shape[2]-1)/2,:][0] != 0):
                cloud_t.append(val[i,:,:,:].flatten())
            else:
                bk_t.append(val[i,:,:,:].flatten())

    rel = np.column_stack((cloud_t,np.ones(len(cloud_t),dtype=int)))
    irrel = np.column_stack((bk_t,np.zeros(len(bk_t),dtype=int)))
    fdata = np.concatenate((rel,irrel),axis=0)
    np.random.shuffle(fdata)
    return(fdata)
     
#%%train_data
fdata = train_data()
vdata = val_data()
tdata = test_data()

#%%
patch_height, patch_width = pn,pn
patch_len = patch_height * patch_width *3
epochs = 25
batch_size =  512

num_classes = 2

input_shape = (patch_width,patch_height,3)
x_train = fdata[:,0:patch_len].reshape(fdata.shape[0], patch_width, patch_height, 3)
y_train = to_categorical(fdata[:,-1],num_classes)

x_val = vdata[:,0:patch_len].reshape(vdata.shape[0], patch_width, patch_height, 3)
y_val = to_categorical(vdata[:,-1],num_classes)

x_test = tdata[:,0:patch_len].reshape(tdata.shape[0], patch_width, patch_height, 3)
y_test = to_categorical(tdata[:,-1],num_classes)

#%%
#Network
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Dense(num_classes, activation='softmax'))

adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy',
              optimizer= adam,
              metrics=['accuracy'])
print(model.summary())
#%%
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val))

score = model.evaluate(x_test,y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#%%predict
for i in range(9,11):    
    qa,b4,b3,b2 = df['BQA'].iloc[i],df['B4'].iloc[i],df['B3'].iloc[i],df['B2'].iloc[i]
    img_t = img_preprocess(qa,b4,b3,b2,tdimn)[0]
    print (img_t.shape)
    with tf.Session() as sess:
        patches3 =  tf.extract_image_patches(images=img_t, ksizes=[1, pn,pn, 1], strides=[1, 1, 1, 1], 
                                             rates=[1, 1, 1, 1], padding='SAME')
        patches3 = tf.reshape(patches3,[-1,pn,pn,3])
        val3 = sess.run(patches3)
        
    d = {0:0,1:1}
    mask_n = np.zeros(tdimn)
    z = 0
    y = 0
    batch = tdimn[0]
    while(y<tdimn[0]):
        pred = model.predict(val3[z:z+batch,:,:,:])
        for i in xrange(tdimn[0]):
            mask_n[i,y] = d[np.argmax(pred[i])]
        z += batch
        y += 1
        if (y % np.sqrt(tdimn[0]) == 0):
            print("No. of iterations till results = {}".format(tdimn[0]-y))
    
    plt.figure(figsize=(40,20))
    plt.subplot(131)
    plt.imshow(np.squeeze(img_t,axis=0),cmap = plt.get_cmap('gist_gray'))
    plt.imshow(mask_n, cmap = plt.get_cmap('Reds'), alpha=0.3)
    plt.title('Cloud-3')
    plt.show()



