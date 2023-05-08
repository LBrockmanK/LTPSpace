import pandas as pd
import numpy as np
from ltpspace import Explorer #,ModelHelper
import h5py
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.utils import shuffle
import os
import logging
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1: INFO, 2: WARNING, 3: ERROR)

def main():
	# Read data
	portion = 100000
	# portion = 4500

	datasets = ['test', 'train', 'valid']

	# Load all datasets into memory
	xdata = []
	ydata = []

	for dataset in datasets:
	    with h5py.File(f'D:\\PCAM Data\\camelyonpatch_level_2_split_{dataset}_x.h5', 'r') as f:
	        xdata.append(f['x'][:portion])
	    with h5py.File(f'D:\\PCAM Data\\camelyonpatch_level_2_split_{dataset}_y.h5', 'r') as f:
	        ydata.append(f['y'][:portion])

	# Concatenate the datasets along the appropriate axis
	x_data_combined = np.concatenate(xdata, axis=0)
	y_data_combined = np.concatenate(ydata, axis=0)

	# Flatten the y_data_combined
	y_data_combined = y_data_combined.flatten()

	# Shuffle the combined dataset
	xdata, ydata = shuffle(x_data_combined, y_data_combined, random_state=42)


	# Set up Explorer
	optimizer = Explorer(samplemodel,xdata,ydata,3,debuglevel = logging.DEBUG,log_directory = "Logfiles")
	optimizer.modify_Model(kfolds = 10, epochs = 10, batch_size = 32)

	# Set up parameters
	optimizer.addParam(18, 18, "Conv2DSize-1")
	optimizer.addParam(128, 128, "Conv2DSize-2")
	optimizer.addParam(976, 976, "Dense1Size")

	# Run CNN model test
	optparams, score = optimizer.run(randomstart = True, recover = True, testonly = True)

	# NOTE: Returning to model v2, seemed to be the best in hindsight, doing a larger data set

# Example model for image classification
def samplemodel(params, input_shape):
		"""Model architecture is based on the VGG16 model but has fewer layers and fewer filters per layer to reduce the model complexity. 
			The VGG16 architecture is known for its good performance in image classification tasks, and it's a good starting point for many 
			classification problems."""
		m = Sequential()

		# First block of convolutional layers
		# Each Conv2D layer performs 2D convolution, which is used for spatial feature extraction.
		# The activation function 'relu' (Rectified Linear Unit) introduces non-linearity and helps to learn complex patterns.
		m.add(Conv2D(params[0], (3, 3), activation='relu', padding='same', input_shape=input_shape))
		# MaxPooling2D layer reduces the spatial dimensions and helps to minimize overfitting by reducing the number of parameters.
		m.add(MaxPooling2D(pool_size=(2, 2)))

		# The activation function 'relu' (Rectified Linear Unit) introduces non-linearity and helps to learn complex patterns.
		m.add(Conv2D(params[1], (3, 3), activation='relu', padding='same', input_shape=input_shape))
		# MaxPooling2D layer reduces the spatial dimensions and helps to minimize overfitting by reducing the number of parameters.
		m.add(MaxPooling2D(pool_size=(2, 2)))

		# Flatten layer is used to convert the 2D feature maps into a 1D vector, which can be used as input to the fully connected layers.
		m.add(Flatten())

		# Fully connected (Dense) layer for classification
		# The first Dense layer with 512 units and 'relu' activation function adds more complexity to the model.
		m.add(Dense(params[2], activation='relu'))

		# Dropout layer helps to prevent overfitting by randomly setting a fraction of input units to 0 during training.
		m.add(Dropout(0.7))

		# The final Dense layer with 1 unit and 'sigmoid' activation function outputs the probability of the image belonging to the positive class.
		m.add(Dense(2, activation='sigmoid'))

		return m

main()