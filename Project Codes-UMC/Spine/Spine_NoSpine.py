# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:20:33 2017

@author: anand
"""

#%%:Header Files and Base Directory
import numpy as np
import SimpleITK as sitk
from os import path
import matplotlib.pyplot as plt
import lasagne
import theano
import cPickle as pickle

#%% load data
T = theano.tensor
L = lasagne.layers

trainS = np.load("trainS.npy")
trainB = np.load("trainB.npy")
testS = np.load("testS.npy")
testB = np.load("testB.npy")


data = T.ftensor4()
labels = T.ivector()


#%% define network
# Define the actual network layer-by-layer


activation = lasagne.nonlinearities.rectify
#w = lasagne.init.HeNormal
#b = lasagne.init.Constant(0.1)

network = L.InputLayer(shape=(None,1,49,49), input_var=data)
network = L.Conv2DLayer(network, num_filters=16, filter_size=3,nonlinearity = activation)
network = L.MaxPool2DLayer(network, pool_size=2)
network = L.Conv2DLayer(network, num_filters=16, filter_size=3,nonlinearity = activation)
network = L.MaxPool2DLayer(network, pool_size=2)
network = L.Conv2DLayer(network, num_filters=16, filter_size=3,nonlinearity = activation)
network = L.Conv2DLayer(network, num_filters=16, filter_size=3,nonlinearity = activation)
network = L.MaxPool2DLayer(network, pool_size=2)
network = L.Conv2DLayer(network, num_filters=16, filter_size=3,nonlinearity = activation)
network = L.dropout(network, p=0.5)
#network = L.DenseLayer(network,num_units = 100, nonlinearity = activation)
network = L.DenseLayer(network, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)

n_params = L.count_params(network, trainable=True)
print('Network defined with {} trainable parameters'.format(n_params))

#%% Objective and sybolic Functions to call the network on training and test data respectively
def objectives(deterministic):
    global network, labels
    predictions = L.get_output(network, deterministic=deterministic)
    
    loss = lasagne.objectives.categorical_crossentropy(predictions, labels).mean()
    loss += 0.0001 * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    
    accuracy = T.mean(T.eq(T.argmax(predictions, axis=1), labels), dtype=theano.config.floatX)
    
    pos = len(np.where(labels == 1.))
    tpos = len(np.where(np.logical_and(np.argmax(predictions,axis=1)==labels, labels == 1.)))
    fpos = len(np.where(np.logical_and(np.argmax(predictions,axis=1) != labels, labels == 1.)))
    dice = tpos/(pos+fpos)
    
    return loss, accuracy, dice

train_loss, train_accuracy = objectives(deterministic=False)
params = L.get_all_params(network, trainable=True)
updates = lasagne.updates.adam(train_loss, params, learning_rate=0.0001)

train = theano.function(inputs=[data, labels], outputs=[train_loss, train_accuracy], 
                        updates=updates, allow_input_downcast=True)
test = theano.function(inputs=[data, labels], outputs=objectives(deterministic=True), 
                       allow_input_downcast=True)

#%% ALL-DATA
#label Spine = 1
#label background = 0


#labels train+test
trainS_l = trainS[:,-1]
trainB_l = trainB[:,-1]
testS_l = testS[:,-1]
testB_l = testB[:,-1]


#Data train+test
trainS = trainS[:,0:trainS.shape[1]-1].reshape((trainS.shape[0],1,49,49))
trainB = trainB[:,0:trainB.shape[1]-1].reshape((trainB.shape[0],1,49,49))
testS = testS[:,0:testS.shape[1]-1].reshape((testS.shape[0],1,49,49))
testB = testB[:,0:testB.shape[1]-1].reshape((testB.shape[0],1,49,49))


#Indices train_test

trainS_i = range(trainS.shape[0])  # for N slices, make a list [0, 1, 2, ..., N-1]
trainB_i = range(trainB.shape[0])
testS_i = range(testS.shape[0])
testB_i= range(testB.shape[0])


#%%train in minbatch
epochs = 50
minibatch_size = 512
def iterate_in_minibatches(name, fn, slices_0, slices_1,labels_0, labels_1, indices_0, indices_1):
    global minibatch_size

    dataset_size = min(len(indices_0), len(indices_1))  # N samples of class 0 and M of class 1 -> get the smaller number

    # Instead of shuffling the actual data (the slices), we just shuffle a list of indices (much more efficient!)
    # If you have enough data, you can also just use random minibatches, so just pick N random samples each time.
    np.random.shuffle(indices_0)
    np.random.shuffle(indices_1)

    performance = []
    
    # Walk from 0 to dataset_size in steps of half a minibatch (because half a minibatch will be 0, the other half 1)
    for mb_start in xrange(0, dataset_size, minibatch_size / 2):
        mb_data = np.concatenate((slices_0[indices_0[mb_start:mb_start+minibatch_size/2],:,:,:],
                                  slices_1[indices_1[mb_start:mb_start+minibatch_size/2],:,:,:]),axis=0)
        
        
        #print mb_data.shape
        mb_labels = np.concatenate((labels_0[indices_0[mb_start:mb_start+minibatch_size/2]],
                                    labels_1[indices_1[mb_start:mb_start+minibatch_size/2]]),axis=0)
        mb_labels = mb_labels.astype(np.int32)
        
        #print mb_labels.shape
        #print mb_labels.dtype
        
        # Turn [xy, xy, n*2] (normal x,y,z format) into [n*2, 1, xy, xy] (theano format, 4D tensor)
        if mb_data.shape[0] != minibatch_size:
            break  # skip incomplete minibatches (shouldn't happen, but just to be sure...)

        
        performance.append(fn(mb_data, mb_labels))

    # We got one value for the loss and one for the accuracy for each minibatch. Let's take the average over all
    # minibatches and display that:
    performance = np.asarray(performance).mean(axis=0)
    print(' > {}: loss = {} ; accuracy = {} ; dice coeff = {}'.format(name, performance[0], performance[1], performance[2]))
    
    return performance

#%%
# First train the network, then test it on the data that was not used for training, then repeat
train_performance = [] 
test_performance = []
for epoch in xrange(1, epochs + 1):
    print('Epoch {}/{}'.format(epoch, epochs))
    train_performance.append(
            iterate_in_minibatches('Training', train, trainS, trainB, trainS_l, trainB_l, trainS_i, trainB_i)
    )
    test_performance.append(
            iterate_in_minibatches('Testing', test, testS, testB, testS_l, testB_l, testS_i, testB_i)
    )

print('Training complete!')

#%% learning curves
def plot_learning_curve(data, color, label):
    data = np.asarray(data)
    epochs = np.arange(data.shape[0])
    
    plt.subplot(121)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epochs, data[:, 0], '-', color=color, label=label)
    plt.axhline(y=0, color='gray', linewidth=1, linestyle='dashed')
    plt.ylim(ymin=-0.05)
    plt.legend(loc='best')
    
    plt.subplot(122)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epochs, data[:, 1], '-', color=color, label=label)
    plt.axhline(y=1, color='gray', linewidth=1, linestyle='dashed')
    plt.ylim(ymin=0, ymax=1.05)
    plt.legend(loc='best')
    
    plt.subplot(123)
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.plot(epochs, data[:, 2], '-', color=color, label=label)
    plt.legend(loc='best')

plt.figure(figsize=(16,8))
plot_learning_curve(train_performance, color='g', label='Train')
plot_learning_curve(test_performance, color='b', label='Test')
plt.show()

np.save('learning_curve_train.npy',train_performance)
np.save('learning_curve_test.npy',test_performance)


#%%predict
classify = theano.function(inputs=[data], outputs=L.get_output(network, deterministic=True), 
                           allow_input_downcast=True)

def read_image(filename):
    image = sitk.ReadImage(filename)  # Use ITK to read the image
    image = sitk.GetArrayFromImage(image)  # Turn ITK image object into a numpy array
    image = np.swapaxes(image, 0, 2)  # ITK returns the image as z,y,x so we flip z<->x to get x,y,z
    return image
basedir = "C:\Users/anand\Documents\NLST\SpineSegmentation_Anand"
thoracicP = read_image(path.join(basedir, 'images/00012_1.3.6.1.4.1.14519.5.2.1.7009.9004.235101717694364072067315843144.mhd'))
thoracicP3 = read_image(path.join(basedir, 'images/00001_1.3.6.1.4.1.14519.5.2.1.7009.9004.208971113691465422967574985259.mhd'))
dims = (thoracicP.shape[0],thoracicP.shape[1],thoracicP.shape[2])
dims3 = (thoracicP3.shape[0],thoracicP3.shape[1],thoracicP3.shape[2])
thoracicM = np.zeros(dims)
thoracicM3 = np.zeros(dims3)


d = {0: 0, 1: 1}
    

pred_data = np.load("pred1.npy")
pred_data = pred_data.reshape((pred_data.shape[0],1,49,49))

pred_data3 = np.load("pred2.npy")
pred_data3 = pred_data3.reshape((pred_data.shape[0],1,49,49))



batch = 512
y = 0
z = 0
while(y < 315):
    pred_mask = classify(pred_data[z:z+batch,:,:,:])
    pred_mask3 = classify(pred_data3[z:z+batch,:,:,:])
    for i  in xrange(512):
        thoracicM[241,i,y] = d[np.argmax(pred_mask[i])]
        thoracicM3[241,i,y] = d[np.argmax(pred_mask3[i])]
    z += batch
    y += 1
    if (y % 15 == 0):
        print("No. of iterations till results = {}".format(315-y))
        
#%%
plt.figure(figsize=(14,12))
plt.imshow(thoracicP[241, :,:].T,cmap = 'gray')
plt.imshow(thoracicM[241, :,:].T, cmap = 'Reds', alpha=0.5)
plt.title('Thoracic1')
plt.show()

plt.figure(figsize=(14,12))
plt.imshow(thoracicP3[241, :,:].T,cmap = 'gray')
plt.imshow(thoracicM3[241, :,:].T, cmap = 'Reds', alpha=0.5)
plt.title('Thoracic3')
plt.show()

#%%
netinfo = {'network': network,'params': L.get_all_param_values(network)}
with open('network.pkl','wb') as f:
    pickle.dump(netinfo,f,protocol = pickle.HIGHEST_PROTOCOL )


