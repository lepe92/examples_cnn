import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

mnist_path=""

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

import theano
#Load the images for training and testing
images, label = load_mnist(dataset="training" , digits=np.arange(10), path=mnist_path)
test_images, test_label = load_mnist(dataset="testing" , digits=np.arange(10), path=mnist_path)
#Flatten the images to get a vectorial representation
images = images.astype(theano.config.floatX).reshape((images.shape[0],1,images.shape[1],images.shape[2]))
test_images = test_images.astype(theano.config.floatX).reshape((test_images.shape[0],1,test_images.shape[1],test_images.shape[2]))
#flatten the labels to get a vectorial representation
label = label.flatten()
test_label = test_label.flatten()
"""
print (test_images.shape)
print (images.shape)
print (label.shape)
print (test_label.shape)
"""
import theano.tensor as T
import theano.tensor.signal 
import theano.tensor.signal.downsample
import numpy as np

learning_rate = T.scalar('lr')
index = T.lscalar('index')
batch_size = 100
train_set_x = theano.shared(value=images, name='train_set_x')
train_set_y = T.cast(theano.shared(value=label, name='train_set_y'), 'int32')

visible_size= (images.shape[2],images.shape[3])
hidden_units=100
output_units=10
rng = np.random.RandomState(1242)
epochs = 10
n_train_batches = images.shape[0]/batch_size
learn_rate = 0.002

########################################
##########INPUTS AND WEIGHTS############
########################################
x = T.tensor4('x')
#output
y = T.ivector('y')
#input weights for the weight matrices are rather important; if they are not set correctly, results might get stuck in local minima
Wc0 = theano.shared(value=np.asarray(rng.uniform(low=-0.035,high=0.035,size=(32,1,3,3)),dtype=theano.config.floatX), name='Wc0')
bc0 = theano.shared(value=np.zeros((32,),dtype=theano.config.floatX), name='bc0')

Wc1 = theano.shared(value=np.asarray(rng.uniform(low=-0.035,high=0.035,size=(64,32,4,4)),dtype=theano.config.floatX), name='Wc1')
bc1 = theano.shared(value=np.zeros((64,),dtype=theano.config.floatX), name='bc1')

#weights first layer
W1 = theano.shared(value=np.asarray(rng.uniform(low=-0.01,high=0.01,size=(64*5*5,hidden_units)),dtype=theano.config.floatX), name='W1')
#bias first layer
b1 = theano.shared(value=np.zeros((hidden_units,),dtype=theano.config.floatX), name='b1')
#weights second layer
W2 = theano.shared(value=np.asarray(rng.uniform(low=-0.05,high=0.05,size=(hidden_units,output_units)),dtype=theano.config.floatX), name='W2')
#bias second layer
b2 = theano.shared(value=np.zeros((output_units,),dtype=theano.config.floatX), name='b2')

########################################
##############  NETWORK  ###############
########################################
conv1Out = T.maximum(0,(T.nnet.conv.conv2d(x, Wc0)+bc0.dimshuffle('x', 0, 'x', 'x')))
maxPool1Out= T.signal.downsample.max_pool_2d(conv1Out, (2,2))
conv2Out = T.maximum(0,(T.nnet.conv.conv2d(maxPool1Out, Wc1)+bc1.dimshuffle('x', 0, 'x', 'x')))
maxPool2Out= T.signal.downsample.max_pool_2d(conv2Out, (2,2))
flattenOut = maxPool2Out.flatten(2)
denselyConnected1Out = T.nnet.sigmoid(T.dot(flattenOut, W1)+b1)
denselyConnected2Out = T.nnet.softmax(T.dot(denselyConnected1Out, W2)+b2)
loss = -T.mean(T.log(denselyConnected2Out)[T.arange(y.shape[0]), y])

prediction = T.argmax(denselyConnected2Out,axis=1)

predict = theano.function([x], prediction)

########################################
#######  GRADIENT AND UPDATES  #########
########################################
params = [Wc0,bc0,Wc1,bc1,W1,b1,W2,b2]
gradient = T.grad(loss, params)
updates = []
for p, g in zip(params, gradient):
   updates.append((p, p - learning_rate * g))

train = theano.function([index, learning_rate], loss, updates=updates,
     givens = {x: train_set_x[(index * batch_size): ((index + 1) * batch_size)],
               y: train_set_y[(index * batch_size): ((index + 1) * batch_size)]})

for epoch in range(epochs):
    # go through training set
    c = []
    for batch_index in range(int(n_train_batches)):
        c.append(train(batch_index, learn_rate))

    predictions_test = predict(test_images)
    accuracy = np.mean(predictions_test == test_label)
    print "epoch",epoch,"| loss", np.mean(c),"| accuracy: %.5f" % accuracy