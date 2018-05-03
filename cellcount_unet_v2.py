# Adapted from https://github.com/zhixuhao/unet
# Data generation adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
# This version is being adapted to have separate modules for data loading, model definition, loading weights etc.

import os

import numpy as np
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import (Conv2D, Cropping2D, Dropout, Input, MaxPooling2D,
                          UpSampling2D, Concatenate)
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error, binary_crossentropy

from random import shuffle

import fileIO
from custom_dataloader import DataGenerator


# Parameters
batch_size = 4

# Rounding errors if dataset has small number of files
training_set_splitfraction = 0.8
nepochs = 3
base_path, _, _, rel_result_path = fileIO.set_paths()[0:4]

# Assign dataset ids based on split fractions
allid = fileIO.get_fileid()
shuffle(allid)  # shuffles the the list in-place
training_partition = allid[0:round(len(allid)*training_set_splitfraction)]
testing_partition = allid[round(len(allid)*training_set_splitfraction):]

# Generators to create training and test sets at runtime
training_generator = DataGenerator(training_partition, batch_size)
validation_generator = DataGenerator(testing_partition, batch_size)

input_im = Input(shape=(512, 512, 1), name='input_im')
conv1 = Conv2D(2, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(input_im)    # (batch, 512, 512, 64)
conv1 = Conv2D(2, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv1)		# (batch, 512, 512, 64)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)				# (batch, 256, 256, 64)
conv2 = Conv2D(4, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(pool1)	    # (batch, 256, 256, 128)
conv2 = Conv2D(4, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv2)	    # (batch, 256, 256, 128)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)			    # (batch, 128, 128, 128)
conv3 = Conv2D(8, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(pool2)		# (batch, 128, 128, 256)
conv3 = Conv2D(8, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv3)		# (batch, 128, 128, 256)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)				# (batch, 64, 64, 256)
conv4 = Conv2D(16, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(pool3)		# (batch, 64, 64, 512)
conv4 = Conv2D(16, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv4)		# (batch, 64, 64, 512)
drop4 = Dropout(0.5)(conv4)									# (batch, 64, 64, 512)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)				# (batch, 32, 32, 512)
conv5 = Conv2D(16, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(pool4)		# (batch, 32, 32, 1024)
conv5 = Conv2D(16, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv5)		# (batch, 32, 32, 1024)
drop5 = Dropout(0.5)(conv5)									# (batch, 32, 32, 1024)
up6 = UpSampling2D(size=(2, 2))(drop5)						# (batch, 64, 64, 1024)
up6 = Conv2D(16, 2, activation='relu', padding='same',
             kernel_initializer='he_normal')(up6)			# (batch, 64, 64, 512)
cat6 = Concatenate(axis=3)([drop4, up6])  # (batch, 64, 64, 1024)
conv6 = Conv2D(16, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(cat6)		# (batch, 64, 64, 512)
conv6 = Conv2D(8, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv6)		# (batch, 64, 64, 512)
up7 = UpSampling2D(size=(2, 2))(conv6)						# (batch, 128, 128, 512)
up7 = Conv2D(8, 2, activation='relu', padding='same',
             kernel_initializer='he_normal')(up7)			# (batch, 128, 128, 256)
cat7 = Concatenate(axis=3)([conv3, up7])				    # (batch, 128, 128, 512)
conv7 = Conv2D(8, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(cat7)		# (batch, 128, 128, 256)
conv7 = Conv2D(8, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv7)		# (batch, 128, 128, 256)
up8 = UpSampling2D(size=(2, 2))(conv7)						# (batch, 256, 256, 256)
up8 = Conv2D(4, 2, activation='relu', padding='same',
             kernel_initializer='he_normal')(up8)			# (batch, 256, 256, 128)
cat8 = Concatenate(axis=3)([conv2, up8])  					# (batch, 256, 256, 256)
conv8 = Conv2D(4, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(cat8)		# (batch, 256, 256, 128)
conv8 = Conv2D(4, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv8)		# (batch, 256, 256, 128)
up9 = UpSampling2D(size=(2, 2))(conv8)						# (batch, 512, 512, 128)
up9 = Conv2D(2, 2, activation='relu', padding='same',
             kernel_initializer='he_normal')(up9)			# (batch, 512, 512, 64)
cat9 = Concatenate(axis=3)([conv1, up9])  				    # (batch, 512, 512, 128)
conv9 = Conv2D(2, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(cat9)	 	# (batch, 512, 512, 64)
conv9 = Conv2D(2, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv9)		# (batch, 512, 512, 64)
conv9 = Conv2D(2, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv9)		# (batch, 512, 512, 2)
output_im = Conv2D(1, 1, activation='sigmoid',
                   name='output_im')(conv9)		            # (batch, 512, 512, 1)

model = Model(inputs=[input_im], outputs=[output_im])

def loss_fcn(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

model.compile(optimizer=Adam(lr=1e-4),
              loss={'output_im':loss_fcn}, metrics=['accuracy'])

model_checkpoint = ModelCheckpoint(
    base_path + rel_result_path + 'unet.h5',
    monitor='loss', verbose=1, save_best_only=True)

history = model.fit_generator(training_generator, epochs=nepochs, verbose=1, callbacks=None, validation_data=validation_generator,
                              max_queue_size=10, workers=1, use_multiprocessing=True, shuffle=True, initial_epoch=0)

im_test = training_generator.x
labels_test = training_generator.y
labels_test_predict = model.predict(im_test, batch_size=1, verbose=1)
im_test = np.reshape(im_test, (np.shape(im_test)[0:3]))
labels_test = np.reshape(labels_test, (np.shape(labels_test)[0:3]))
labels_test_predict = np.reshape(
    labels_test_predict, (np.shape(labels_test_predict)[0:3]))

import im3dscroll as I
I.im3dscroll(im_test)
I.im3dscroll(labels_test)
I.im3dscroll(labels_test_predict)
