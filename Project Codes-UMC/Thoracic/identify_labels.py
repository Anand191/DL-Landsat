# -*- coding: utf-8 -*-
"""
Created on Wed May 24 13:31:51 2017

@author: anand
"""
#%%
import numpy as np
import SimpleITK as sitk
from os import path
import matplotlib.pyplot as plt

from patches import PatchExtractor3D



basedir = "C:\Users/anand\Documents\NLST\SpineSegmentation_Anand"

#%%
def read_image(filename):
    image = sitk.ReadImage(filename)  # Use ITK to read the image
    image = sitk.GetArrayFromImage(image)  # Turn ITK image object into a numpy array
    image = np.swapaxes(image, 0, 2)  # ITK returns the image as z,y,x so we flip z<->x to get x,y,z
    return image

thoracic_img = read_image(path.join(basedir, 'images/00012_1.3.6.1.4.1.14519.5.2.1.7009.9004.235101717694364072067315843144.mhd'))
thoracic_mask = read_image(path.join(basedir, 'masks/00012_1.3.6.1.4.1.14519.5.2.1.7009.9004.235101717694364072067315843144.mhd'))

#%%
plt.figure(figsize=(14,12))
plt.imshow(thoracic_img[235, :,:].T,cmap = 'gray')
plt.imshow(thoracic_mask[235, :,:].T, cmap = 'Reds', alpha=0.5)
plt.title('Thoracic1')
plt.show()

#%%
labels = np.where(thoracic_mask[234,:,:] != 0)

def training_set():
    vertebrae = []
    background = []
    train_slices = [240,238,236,234]
    
    extractor = PatchExtractor3D(thoracic_img, pad_value=-1000)
    for s in train_slices:
        i = 0
        while(i<315):
            for j in xrange(0,512):
                patch = extractor.extract_rect(center_voxel=(s, j, i), shape=(49, 49), axis=0)
                if(thoracic_mask[s,j,i] != 0):
                    vertebrae.append(patch.flatten().tolist()+[thoracic_mask[s,j,i]])
                else:
                    background.append(patch.flatten().tolist()+[0])
            i+=1
            
    #print(vertebrae[1],background[10])
            
    return (np.asarray(vertebrae),np.asarray(background))

train_data = training_set()

#%%
np.save('trainV.npy',train_data[0])
np.save('trainBG.npy',train_data[1])
#%%
def test_set():
    vertebrae = []
    background = []
    test_slices = [239,237]
    extractor = PatchExtractor3D(thoracic_img, pad_value=-1000)
    for s in test_slices:
        i = 0
        while(i<315):
            for j in xrange(0,512):
                patch = extractor.extract_rect(center_voxel=(s, j, i), shape=(49, 49), axis=0)
                if(thoracic_mask[s,j,i] != 0):
                    vertebrae.append(patch.flatten().tolist()+[thoracic_mask[s,j,i]])
                else:
                    background.append(patch.flatten().tolist()+[0])
            i+=1
            
    #print(vertebrae[1],background[10])
            
    return (np.asarray(vertebrae),np.asarray(background))

test_data = test_set()

#%%
np.save('testV.npy',test_data[0])
np.save('testBG.npy',test_data[1])

#%%
def pred_set():
    pred = []
    extractor = PatchExtractor3D(thoracic_img, pad_value=-1000)
    i = 0
    while(i<315):
        for j in xrange(0,512):
            patch = extractor.extract_rect(center_voxel=(235, j, i), shape=(49, 49), axis=0)
            pred.append(patch.flatten().tolist())
        i+=1
            
    #print(vertebrae[1],background[10])
            
    return (np.asarray(pred))

pred_data = pred_set()
np.save('pred.npy',pred_data)
