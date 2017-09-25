# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 13:39:23 2017

@author: anand
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:20:33 2017

@author: anand
"""

#%%:Header Files and Base Directory
import numpy as np
import matplotlib.pyplot as plt
import lasagne
import theano
import cPickle as pickle

#%% load data
T = theano.tensor
L = lasagne.layers

trainS = np.load("trainV.npy")
trainB = np.load("trainBG.npy")
testS = np.load("testV.npy")
testB = np.load("testBG.npy")


data = T.ftensor4()
labels = T.ivector()


#%% define network
# Define the actual network layer-by-layer


activation = lasagne.nonlinearities.rectify
w = lasagne.init.GlorotNormal('relu')
B = lasagne.init.Constant(0.001)

network = L.InputLayer(shape=(None,1,49,49), input_var=data)

network = L.Conv2DLayer(network, num_filters=16, filter_size=3,nonlinearity = activation, W = w, b = B)
network = L.Conv2DLayer(network, num_filters=32, filter_size=3,nonlinearity = activation, W = w, b = B)
network = L.MaxPool2DLayer(network, pool_size=2)

network = L.Conv2DLayer(network, num_filters=32, filter_size=3,nonlinearity = activation, W = w, b = B)
network = L.Conv2DLayer(network, num_filters=32, filter_size=3,nonlinearity = activation, W = w, b = B)
network = L.MaxPool2DLayer(network, pool_size=2)

network = L.Conv2DLayer(network, num_filters=64, filter_size=3,nonlinearity = activation, W = w, b = B)
network = L.Conv2DLayer(network, num_filters=64, filter_size=3,nonlinearity = activation, W = w, b = B)
network = L.Conv2DLayer(network, num_filters=64, filter_size=3,nonlinearity = activation, W = w, b = B)
network = L.MaxPool2DLayer(network, pool_size=3)

network = L.Conv2DLayer(network, num_filters=128, filter_size=1,nonlinearity = activation)

network = L.dropout(network, p=0.5)

network = L.DenseLayer(network, num_units=12, nonlinearity=lasagne.nonlinearities.softmax)

n_params = L.count_params(network, trainable=True)
print('Network defined with {} trainable parameters'.format(n_params))

#%% Objective and sybolic Functions to call the network on training and test data respectively
def objectives(deterministic):
    global network, labels
    predictions = L.get_output(network, deterministic=deterministic)
    
    #print(predictions,labels)
    
    loss = lasagne.objectives.categorical_crossentropy(predictions, labels).mean()
    loss += 0.0001 * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    
    accuracy = T.mean(T.eq(T.argmax(predictions, axis=1), labels), dtype=theano.config.floatX)
    #accuracy = T.mean(lasagne.objectives.categorical_accuracy(predictions, labels, top_k=1))
    return loss, accuracy

train_loss, train_accuracy = objectives(deterministic=False)
params = L.get_all_params(network, trainable=True)
updates = lasagne.updates.sgd(train_loss, params, learning_rate=0.0001)

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
epochs = 10
minibatch_size = 252
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
        mb_data = np.concatenate((slices_0[indices_0[mb_start:mb_start+(11*minibatch_size/12)],:,:,:],
                                  slices_1[indices_1[mb_start:mb_start+minibatch_size/12],:,:,:]),axis=0)
        
        
        #print mb_data.shape
        mb_labels = np.concatenate((labels_0[indices_0[mb_start:mb_start+(11*minibatch_size/12)]],
                                    labels_1[indices_1[mb_start:mb_start+minibatch_size/12]]),axis=0)
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
    print(' > {}: loss = {} ; accuracy = {}'.format(name, performance[0], performance[1]))
    
    return performance

#%%
# First train the network, then test it on the data that was not used for training, then repeat
for epoch in xrange(1, epochs + 1):
    print('Epoch {}/{}'.format(epoch, epochs))
    iterate_in_minibatches('Training', train, trainS, trainB, trainS_l, trainB_l, trainS_i, trainB_i)
    iterate_in_minibatches('Testing', test, testS, testB, testS_l, testB_l, testS_i, testB_i)

print('Training complete!')

#%%

netinfo = {'network': network,'params': L.get_all_param_values(network)}
f = open('network.pkl','wb')
pickle.dump(netinfo,f,protocol = pickle.HIGHEST_PROTOCOL )

