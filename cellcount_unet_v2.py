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
import pdb

# Parameters
batch_size = 4

# Rounding errors if dataset has small number of files
training_set_splitfraction = 0.8
nepochs = 50
base_path, _, _, rel_result_path = fileIO.set_paths()[0:4]

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

input_im = Input(shape=(256, 256, 1), name='input_im')
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


model.compile(optimizer=Adam(),loss={'output_im': loss_fcn}, metrics=['accuracy'])

model_checkpoint = ModelCheckpoint(
    base_path + rel_result_path + 'unet.h5',
    monitor='loss', verbose=1, save_best_only=True)

#----------------------------------------------
#----------------------------------------------
#----------------------------------------------

#         checkpoint_cb = ModelCheckpoint(filepath=(checkpoint_path + '/' + '{epoch:04d}' + '.h5'),
#                                     verbose=1, save_best_only=False, save_weights_only=True,
#                                     mode='auto', period=save_period)

#     train_history = autoencoder.fit({'input_x': x_train},
#                                     {'Rx1': x_train,
#                                      'Rx2': x_train,
#                                      'z1': np.zeros((train_size, bottleneck_dim)),
#                                      'z2': np.zeros((train_size, bottleneck_dim))},
#                                     epochs=n_epoch,
#                                     batch_size=batch_size,
#                                     shuffle=True,
#                                     validation_data=({'input_x': x_test},
#                                                      {'Rx1': x_test,
#                                                       'Rx2': x_test,
#                                                       'z1': np.zeros((test_size, bottleneck_dim)),
#                                                       'z2': np.zeros((test_size, bottleneck_dim))}),
#                                     callbacks=[checkpoint_cb])

#     summary = train_history.params
#     summary.update(train_history.history)

# # Save trained model
#     plot_model(autoencoder, to_file=save_path + fileid+'-img'+'.png')
#     autoencoder.save_weights(save_path+fileid+'-modelweights'+'.h5')
#     with open(save_path+fileid+'-traininghist'+'.pkl', 'wb') as file_pi:
#         cPickle.dump(summary, file_pi)

# #----------------------------------------------
# #----------------------------------------------
# #----------------------------------------------

#history = model.fit({'input_im': training_generator.x}, {'output_im': training_generator.y}, batch_size=4, epochs=nepochs, verbose=1,
#                    validation_data=({'input_im': training_generator.x}, {'output_im': training_generator.y}), shuffle=True)

history = model.fit_generator(training_generator, epochs=nepochs, verbose=1, callbacks=None, validation_data=validation_generator,
                              max_queue_size=10, workers=1, use_multiprocessing=True, shuffle=True, initial_epoch=0)

im_test = validation_generator.x
labels_test = validation_generator.y
labels_test_predict = model.predict(im_test, batch_size=1, verbose=1)
im_test = np.reshape(im_test, (np.shape(im_test)[0:3]))
labels_test = np.reshape(labels_test, (np.shape(labels_test)[0:3]))
labels_test_predict = np.reshape(
    labels_test_predict, (np.shape(labels_test_predict)[0:3]))

import im3dscroll as I
I.im3dscroll(im_test)
I.im3dscroll(labels_test)
I.im3dscroll(labels_test_predict)

import matplotlib.pyplot as plt
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(history.history['loss'])
plt.show(block=True)
