#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:51:31 2017

@author: anand
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:59:22 2017

@author: anand
"""

import os
import pandas as pd
from pymasker import LandsatMasker
from pymasker import LandsatConfidence
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from U_Net3 import myUnet

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
np.random.shuffle(idx)
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

#%%
def create_test_data(idx2):
    imgdatas = np.ndarray((len(idx2),tdimn[0],tdimn[1],3))
    imgmasks = np.ndarray((len(idx2),tdimn[0],tdimn[1],1))
    j = 0
    for i in idx2:
        qa = df['BQA'].iloc[i]
        imgs_test, imgs_mask_test = img_preprocess(i,qa)
        imgdatas[j] = imgs_test
        imgmasks[j] = imgs_mask_test
        j += 1
    imgdatas = imgdatas.astype('float32')
    imgmasks = imgmasks.astype('float32')
    imgdatas/= 255.
    
    imgmasks /= 255.
    imgmasks[imgmasks > 0.5] = 1
    imgmasks[imgmasks <= 0.5] = 0    

    print('completed loading test data')
    return (imgdatas,imgmasks)  
#%%
imgs_test,imgs_mask_test_a = create_test_data(idx[16:])
myunet = myUnet()
model = myunet.get_unet()

#%%
model.load_weights('unet_v2.h5')
imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)


#%%
for j in xrange(imgs_test.shape[0]):                    
    plt.figure(figsize=(40,20))
    plt.subplot(131)
    plt.imshow(imgs_test[j],cmap=plt.get_cmap('gist_gray'))
    plt.imshow(np.squeeze(imgs_mask_test[j],axis=2),cmap=plt.get_cmap('Reds'),alpha=0.3)    

#%%
imgs_mask_test[imgs_mask_test > 0.5] = 1
imgs_mask_test[imgs_mask_test <= 0.5] = 0

for j in xrange(imgs_test.shape[0]):  
    plt.figure(figsize=(40,20))
    plt.subplot(131)
    plt.imshow(imgs_test[j])
    plt.imshow(np.squeeze(imgs_mask_test[j],axis=2),cmap=plt.get_cmap('Reds'),alpha=0.3)
#%%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools

y_true = np.ravel(np.squeeze(imgs_mask_test_a,axis=3))
y_pred = np.ravel(np.squeeze(imgs_mask_test,axis=3))

#%%
labels = [0,1]
target_names = ['Earth','Cloud']

cm = confusion_matrix(y_true,y_pred,labels)
print("confusion matrix=")    
print(cm)

print(classification_report(y_true, y_pred, target_names=target_names))
fig1 = plt.figure(figsize=(12,10))
ax = fig1.add_subplot(211)
cax = ax.matshow(cm,cmap=plt.cm.Blues)
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    ax.text(j,i,cm[i,j])
plt.title('Confusion matrix of UNet')
fig1.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# =============================================================================
# for k in range(imgs_mask_test.shape[0]):
#     cm = confusion_matrix(np.ndarray.flatten(np.squeeze(imgs_mask_test_a[k],axis=2)),
#                           np.ndarray.flatten(np.squeeze(imgs_mask_test[k],axis=2)),labels)
#     print("confusion matrix=")    
#     print(cm)
#     
#     
#     
#     fig1 = plt.figure(figsize=(12,10))
#     ax = fig1.add_subplot(211)
#     cax = ax.matshow(cm)
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         ax.text(j,i,cm[i,j])
#     plt.title('Confusion matrix of the classifier')
#     fig1.colorbar(cax)
#     ax.set_xticklabels([''] + labels)
#     ax.set_yticklabels([''] + labels)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.show()
# =============================================================================

