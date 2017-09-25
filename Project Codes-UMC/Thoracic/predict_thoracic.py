# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 11:37:20 2017

@author: anand
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import lasagne
import theano
import cPickle as pickle
import SimpleITK as sitk
from os import path
from patches import PatchExtractor3D

#%%
T = theano.tensor
L = lasagne.layers
data = T.ftensor4()
labels = T.ivector()

activation = lasagne.nonlinearities.rectify
w = lasagne.init.GlorotNormal('relu')
B = lasagne.init.Constant(0.01)

network = L.InputLayer(shape=(None,1,49,49), input_var=data)

network = L.Conv2DLayer(network, num_filters=16, filter_size=3,nonlinearity = activation, W = w, b = B)
network = L.Conv2DLayer(network, num_filters=32, filter_size=3,nonlinearity = activation, W = w, b = B)
network = L.MaxPool2DLayer(network, pool_size=2)

network = L.Conv2DLayer(network, num_filters=32, filter_size=3,nonlinearity = activation, W = w, b = B)
network = L.Conv2DLayer(network, num_filters=64, filter_size=3,nonlinearity = activation, W = w, b = B)
network = L.MaxPool2DLayer(network, pool_size=2)

network = L.Conv2DLayer(network, num_filters=64, filter_size=3,nonlinearity = activation, W = w, b = B)
network = L.Conv2DLayer(network, num_filters=128, filter_size=3,nonlinearity = activation, W = w, b = B)
network = L.Conv2DLayer(network, num_filters=256, filter_size=3,nonlinearity = activation, W = w, b = B)
network = L.MaxPool2DLayer(network, pool_size=3)

network = L.Conv2DLayer(network, num_filters=512, filter_size=1,nonlinearity = activation)

network = L.dropout(network, p=0.5)

network = L.DenseLayer(network, num_units=12, nonlinearity=lasagne.nonlinearities.softmax)

n_params = L.count_params(network, trainable=True)
print('Network defined with {} trainable parameters'.format(n_params))

#%%predict
net = pickle.load(open('network.pkl','rb'))
all_params = net['params']
lasagne.layers.set_all_param_values(network,all_params)

classify = theano.function(inputs=[data], outputs=L.get_output(network, deterministic=True), 
                           allow_input_downcast=True)

def read_image(filename):
    image = sitk.ReadImage(filename)  # Use ITK to read the image
    image = sitk.GetArrayFromImage(image)  # Turn ITK image object into a numpy array
    image = np.swapaxes(image, 0, 2)  # ITK returns the image as z,y,x so we flip z<->x to get x,y,z
    return image
basedir = "C:\Users/anand\Documents\NLST\SpineSegmentation_Anand"
thoracicP = read_image(path.join(basedir, 'images/00012_1.3.6.1.4.1.14519.5.2.1.7009.9004.235101717694364072067315843144.mhd'))
dims = (thoracicP.shape[0],thoracicP.shape[1],thoracicP.shape[2])
thoracicM = np.zeros(dims)

class_labels = np.append(np.linspace(19,9,11),[0])
keys = np.arange(12)
d = {}
for x in xrange(12):
    d.update({keys[x]:int(class_labels[x])})
    
extractor = PatchExtractor3D(thoracicP, pad_value=-1000)
pred_data = np.load("pred.npy")
pred_data = pred_data.reshape((pred_data.shape[0],1,49,49))
#pred_mask = classify(pred_data_n)
batch = 512
y = 0
z = 0
while(y < 315):
    pred_mask = classify(pred_data[z:z+batch,:,:,:])
    for i  in xrange(512):
        thoracicM[235,i,y] = d[np.argmax(pred_mask[i])]
    z += batch
    y += 1
    if (y % 15 == 0):
        print("No. of iterations till results = {}".format(315-y))
#==============================================================================
# y = 0    
# while(y<315):
#     for z in xrange(0,512):
#         patch = extractor.extract_rect(center_voxel=(235, z, y), shape=(49, 49), axis=0)
#         pred_data = patch.reshape((1,1,49,49))
#         pred_mask = classify(pred_data)
#         thoracicM[235,z,y] = d[np.argmax(pred_mask[0])]
#     print ('test')
#     y += 1
#==============================================================================

#%%
plt.figure(figsize=(14,12))
plt.imshow(thoracicP[235, :,:].T,cmap = 'gray')
plt.imshow(thoracicM[235, :,:].T, cmap = 'Reds', alpha=0.5)
plt.title('Thoracic1')
plt.show()
