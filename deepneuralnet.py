# Author: Scott Gordon
# Modified by: Ajinkya Malhotra
#
# Model that defines all the architecture used for the neural network

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy

acc = Accuracy()
network = input_data(shape=[None, 100, 100, 1])

# Conv layers ------------------------------------      
network = conv_2d(network, 64, 3, strides=1, activation='tanh')
network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 64, 3, strides=1, activation='tanh')
network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 64, 3, strides=1, activation='tanh')
network = conv_2d(network, 64, 3, strides=1, activation='relu')
network = conv_2d(network, 64, 3, strides=1, activation='relu')
network = max_pool_2d(network, 2, strides=2)

# Fully Connected Layers -------------------------
network = fully_connected(network, 1024, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 1024, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 7, activation='softmax')
network = regression(network, optimizer='momentum',
loss='categorical_crossentropy',
learning_rate=0.003, metric=acc)
model = tflearn.DNN(network)
