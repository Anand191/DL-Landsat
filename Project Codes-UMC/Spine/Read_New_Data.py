# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:00:21 2017

@author: anand
"""

#%%:Header Files and Base Directory
import numpy as np
import SimpleITK as sitk
from os import path
import matplotlib.pyplot as plt
from patches import PatchExtractor3D

basedir = "C:\Users/anand\Documents\NLST\SpineSegmentation_Anand"

#%%: Read Data
def read_image(filename):
    image = sitk.ReadImage(filename)  # Use ITK to read the image
    image = sitk.GetArrayFromImage(image)  # Turn ITK image object into a numpy array
    image = np.swapaxes(image, 0, 2)  # ITK returns the image as z,y,x so we flip z<->x to get x,y,z
    return image

thoracic1 = read_image(path.join(basedir, 'images/00012_1.3.6.1.4.1.14519.5.2.1.7009.9004.235101717694364072067315843144.mhd'))
thoracic2 = read_image(path.join(basedir, 'masks/00012_1.3.6.1.4.1.14519.5.2.1.7009.9004.235101717694364072067315843144.mhd'))
thoracic3 = read_image(path.join(basedir, 'images/00001_1.3.6.1.4.1.14519.5.2.1.7009.9004.208971113691465422967574985259.mhd'))
# Let's print out the shape of the numpy array "onback" (which should be a 3D matrix 96x96xN with N = number of slices)
print(thoracic1.shape)
print(thoracic2.shape)
print(thoracic3.shape)


#%%: Show Scans
plt.figure(figsize=(40,20))
plt.subplot(131)
plt.imshow(thoracic1[240, :,:].T,cmap = plt.cm.gray)
plt.imshow(thoracic2[240, :,:].T, cmap = plt.get_cmap('Reds'), alpha=0.5)
plt.title('Thoracic1')
plt.show()

#%% Create Training Set
def training_set():
    spine = []
    background = []
    train_slices = [234,235,236,237,238]
    factor = [5,10]
    extractor = PatchExtractor3D(thoracic1, pad_value=-1000)
    for s in train_slices:        
        i = 0
        while(i<315):
            for j in range(0,512):
                rot_angle = np.multiply(factor,list(np.random.normal(size=2)))
                np.append(rot_angle,0.)
                for angle in rot_angle:
                    if(angle != 0.0):                        
                        patch = extractor.extract_rect(center_voxel=(s, j, i), shape=(49, 49), axis=0, rotation_angle = angle)
                        if(thoracic2[s, j, i] > 0):
                            spine.append(patch.flatten())
                        else:
                            background.append(patch.flatten())
                    elif(angle==0.0):
                        patch = extractor.extract_rect(center_voxel=(s, j, i), shape=(49, 49), axis=0)
                        if(thoracic2[s, j, i] > 0):
                            spine.append(patch.flatten())
                        else:
                            background.append(patch.flatten())
            i+=1
    sp = np.column_stack((spine,np.ones(len(spine))))
    bk = np.column_stack((background,np.zeros(len(background))))
    return (sp,bk)
print("train file writing started")
train_data = training_set()


np.save('trainS.npy',train_data[0])
np.save('trainB.npy',train_data[1])

print("train file writing completed")

#%%Create Test
def test_set():
    test_spine = []
    test_background = []
    test_slices = [239,240]
    factor = [5,10]
    extractor = PatchExtractor3D(thoracic1, pad_value=-1000)
    for s in test_slices:        
        i = 0
        while(i<315):
            for j in range(0,512):
                rot_angle = np.multiply(factor,list(np.random.normal(size=2)))
                np.append(rot_angle,0.)
                for angle in rot_angle:
                    if(angle != 0.0):                        
                        patch = extractor.extract_rect(center_voxel=(s, j, i), shape=(49, 49), axis=0, rotation_angle = angle)
                        if(thoracic2[s, j, i] > 0):
                            test_spine.append(patch.flatten())
                        else:
                            test_background.append(patch.flatten())
                    elif(angle==0.0):
                        patch = extractor.extract_rect(center_voxel=(s, j, i), shape=(49, 49), axis=0)
                        if(thoracic2[s, j, i] > 0):
                            test_spine.append(patch.flatten())
                        else:
                            test_background.append(patch.flatten())
            i+=1
    tsp = np.column_stack((test_spine,np.ones(len(test_spine))))
    tbk = np.column_stack((test_background,np.zeros(len(test_background))))
    return (tsp,tbk)

print("test file writing started")
test_data = test_set()

np.save('testS.npy',test_data[0])
np.save('testB.npy',test_data[1])

print("testfile writing completed")

#%% Create Prediction
def pred_set():
    pred = []
    predict_set = [thoracic1, thoracic3]
    z = 0
    for p in predict_set:
        extractor = PatchExtractor3D(p, pad_value=-1000)
        z += 1        
        i = 0
        while(i<315):
            for j in xrange(0,512):
                patch = extractor.extract_rect(center_voxel=(241, j, i), shape=(49, 49), axis=0)
                pred.append(patch.flatten().tolist())
            i+=1
        np.save('pred%s.npy'%z,np.asarray(pred))
            
    print("finished file writing")
            
     
print("prediction file writing started")
pred_set()



