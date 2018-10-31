# Author: Scott Gordon
# Modified by: Ajinkya Malhotra
#
# Runs the trained model, to validate the testing dataset

import deepneuralnet as net
import numpy as np
from tflearn.data_utils import image_preloader

model = net.model

#Saves the filepath of the trained model
path_to_model = './ZtrainedNet/final-model.tfl'

#Load's the trained model
model.load(path_to_model)

#Loads images on the fly from the validate directory
X, Y = image_preloader(target_path='./validate', image_shape=(100,100), mode='folder', grayscale=True, categorical_labels=True, normalize=True)

#Refactors the incoming layer tensor output to a 100x100x1 size
X = np.reshape(X, (-1, 100, 100, 1))

for i in range(0, len(X)):
 iimage = X[i]
 icateg = Y[i]
 result = model.predict([iimage])[0]
 prediction = result.tolist().index(max(result))
 reality = icateg.tolist().index(max(icateg))

#if true then
 if prediction == reality:
    print("image %d CORRECT " % i, end='')

#if false then
 else:
    print("image %d WRONG " % i, end='')

#Prints the result
 print(result)
