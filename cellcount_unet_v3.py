# Adapted from https://github.com/zhixuhao/unet
# Data generation ideas: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
# specify runmode, runid (withoutthe archid) and epochid to load model weights
# Architecture requires a 64 x 64 input.

import os
import pdb
import sys
import time
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import (Concatenate, Conv2D, Cropping2D, Dropout, Input,
                          MaxPooling2D, UpSampling2D)
from keras.losses import binary_crossentropy, mean_squared_error
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

import _pickle as cPickle
import pickle as Pickle
import fileIO
import im3dscroll as I
from custom_dataloader import DataGenerator
import tensorflow as tf

# Parameters
batch_size = 64

# Rounding errors if dataset has small number of files
archid = '_v3'
nepochs = 200
save_period = 50
training_set_splitfraction = 0.8
if len(sys.argv) > 1:
    runid = sys.argv[1] + archid
else:
    runid = time.strftime("%Y%m%d-%H%M%S") + archid

runmode = 'trainnew' #'trainnew' or 'viewresult' 
epochid = '0700' #only used if runmode is 'viewresult'


base_path, _, _, rel_result_path = fileIO.set_paths()[0:4]
checkpoint_path = base_path + rel_result_path + '/' + runid
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)


# Assign dataset ids based on split fractions
allid = fileIO.get_fileid()
shuffle(allid)  # shuffles the the list in-place
training_partition = allid[0:round(len(allid)*training_set_splitfraction)]
testing_partition = allid[round(len(allid)*training_set_splitfraction):]

training_partition = ['123','77','60','57','167','68','113','157','143','109','175','95','119','74','131','111']
testing_partition = ['53']

# Generators to create training and test sets at runtime
training_generator = DataGenerator(training_partition, batch_size)
validation_generator = DataGenerator(testing_partition, batch_size)

input_im = Input(shape=(64, 64, 1), name='input_im')
conv2 = Conv2D(4, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(input_im)    # (batch, 256, 256, 128)
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
conv9 = Conv2D(2, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv8)		# (batch, 256, 256, 2)
output_im = Conv2D(1, 1, activation='sigmoid',
                   name='output_im')(conv9)		            # (batch, 256, 256, 1)

model = Model(inputs=[input_im], outputs=[output_im])


def loss_fcn(y_true, y_pred):
    w = y_true
    # based on number of non-zero labels in the image
    w = (548655.0/111411200.0)*(1 - y_true) + (110862545.0/111411200.0)*(y_true)
    return K.mean(K.square(y_true - y_pred)*w, axis=None)

def loss_fcn_bce(y_true, y_pred):
    weight_ones = 0.995 #~110862545.0/111411200.0
    weight_zeros = 1. - weight_ones
    #wbce = - tf.multiply(weight_ones, tf.multiply(y_true, K.log(y_pred))) \
    #       - tf.multiply(weight_zeros, tf.multiply((1. - y_true), K.log(1. - y_pred)))
    wbce = - (weight_ones * y_true * K.log(y_pred + K.epsilon())) \
           - (weight_zeros * (1.-y_true) * K.log(1. - y_pred + K.epsilon()))
    return K.mean(wbce,axis = None)


model.compile(optimizer=Adam(),loss={'output_im': loss_fcn_bce}, metrics=['accuracy'])

bestnet_cb = ModelCheckpoint(filepath=(base_path + rel_result_path + runid + '-bestwt.h5'),
                             monitor='loss', verbose=1, save_best_only=True)

hist_cb = ModelCheckpoint(filepath=(base_path + rel_result_path + '/' + runid + '/' + '{epoch:04d}' + '.h5'),
                          verbose=1, save_best_only=False, save_weights_only=True,
                          mode='auto', period=save_period)

#To train without using generator:
#history = model.fit({'input_im': training_generator.x}, {'output_im': training_generator.y}, batch_size=4, epochs=nepochs, verbose=1,
#                    validation_data=({'input_im': training_generator.x}, {'output_im': training_generator.y}), shuffle=True)
if runmode == 'trainnew':
    train_history = model.fit_generator(training_generator, epochs=nepochs, verbose=1, callbacks=[hist_cb], validation_data=validation_generator,
                                        max_queue_size=10, workers=1, use_multiprocessing=True, shuffle=True, initial_epoch=0)
    summary = train_history.params
    summary.update(train_history.history)

    # Save trained model
    plot_model(model, to_file=base_path +
               rel_result_path + runid + '-arch'+'.png')
    with open(base_path + rel_result_path + runid+'-summary'+'.pkl', 'wb') as file_pi:
        cPickle.dump(summary, file_pi)

elif runmode == 'viewresult':
    model.load_weights(base_path + rel_result_path + '/' + runid + '/' + epochid + '.h5')
    with open(base_path + rel_result_path + runid+'-summary'+'.pkl', 'rb') as file_pi:
        summary = Pickle.load(file_pi)

im_test = validation_generator.x
labels_test = validation_generator.y
labels_test_predict = model.predict(im_test, batch_size=1, verbose=1)
im_test = np.reshape(im_test, (np.shape(im_test)[0:3]))
labels_test = np.reshape(labels_test, (np.shape(labels_test)[0:3]))
labels_test_predict = np.reshape(
    labels_test_predict, (np.shape(labels_test_predict)[0:3]))

I.im3dscroll(im_test)
I.im3dscroll(labels_test)
I.im3dscroll(labels_test_predict)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(summary['loss'])
plt.show(block=False)