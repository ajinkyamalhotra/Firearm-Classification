# Author: Scott Gordon
# Modified by: Ajinkya Malhotra
#
# Trains the model based on the training datasets
# Saves the trained model as final-model.tfl

import deepneuralnet as net
import numpy as np
from tflearn.data_utils import image_preloader

model = net.model

#Loads images on the fly from the train directory
X, Y = image_preloader(target_path='./train', image_shape=(100, 100), mode='folder', grayscale=True, categorical_labels=True, normalize=True)

#Refactors the incoming layer tensor output to a 100x100x1 size
X = np.reshape(X, (-1, 100, 100, 1))

#Loads images on the fly from the validate directory
W, Z = image_preloader(target_path='./validate', image_shape=(100, 100), mode='folder', grayscale=True, categorical_labels=True, normalize=True)

#Refactors the incoming layer tensor output to a 100x100x1 size
W = np.reshape(W, (-1, 100, 100, 1))

#Initializes the model to run, runs for 1000 epochs and processes 500 images at a time
model.fit(X, Y, n_epoch=1000, batch_size = 500,validation_set=(W,Z), show_metric=True)

#Saves the trained model to ZtrainedNet directory as final-model.tfl
model.save('./ZtrainedNet/final-model.tfl')
