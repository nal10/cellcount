# Adapted from https://github.com/zhixuhao/unet

import os

import numpy as np
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import (Conv2D, Cropping2D, Dropout, Input, MaxPooling2D,
                          UpSampling2D, Concatenate)
from keras.models import Model
from keras.optimizers import Adam

import fileIO
import pdb

base_path,_,_,rel_result_path = fileIO.set_paths()[0:4]

# return im_train, im_train_labels, im_test, im_test_labels
im = fileIO.load_IM()
labels = fileIO.load_labels()

# Below block puts patches from different images into different list elements
im_train_list = []
labels_train_list = []
for i in range(0, im.shape[0]):
    im_train_list += [fileIO.gen_patch(im[i], stride=(256, 256))]
    labels_train_list += [fileIO.gen_patch(labels[i], stride=(256, 256))]

del im
del labels

im_train = fileIO.stack_list(im_train_list)
del im_train_list

im_train_labels = fileIO.stack_list(labels_train_list)
del labels_train_list

# Split data explicitly into training and test sets
validation_split = 0.1
test_inds = np.arange(np.size(im_train, 0))
test_inds = test_inds[test_inds <= np.size(im_train, 0)*validation_split]

im_test = im_train[test_inds]
im_test_labels = im_train_labels[test_inds]

im_train = np.delete(im_train, test_inds, axis=0)
im_train_labels = np.delete(im_train_labels, test_inds, axis=0)

#Conversion to float32. I'm guessing this saves some memory?
im_train = im_train.astype('float32') / 255.
im_train_labels = im_train_labels.astype('float32') / 255.
im_test = im_test.astype('float32') / 255.
im_test_labels = im_test_labels.astype('float32') / 255.

#Reshape to add the 4th dimension
im_train = np.reshape(im_train, (len(im_train), 512, 512, 1))
im_train_labels = np.reshape(im_train_labels, (len(im_train_labels), 512, 512, 1))
im_test = np.reshape(im_test, (len(im_test), 512, 512, 1))
im_test_labels = np.reshape(im_test_labels, (len(im_test_labels), 512, 512, 1))

inputs = Input(shape=(512, 512, 1), name='inputs')
conv1 = Conv2D(64, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(inputs)		# (batch, 512, 512, 64)
conv1 = Conv2D(64, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv1)		# (batch, 512, 512, 64)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)				# (batch, 256, 256, 64)
conv2 = Conv2D(128, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(pool1)	    # (batch, 256, 256, 128)
conv2 = Conv2D(128, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv2)	    # (batch, 256, 256, 128)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)			    # (batch, 128, 128, 128)
conv3 = Conv2D(256, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(pool2)		# (batch, 128, 128, 256)
conv3 = Conv2D(256, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv3)		# (batch, 128, 128, 256)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)				# (batch, 64, 64, 256)
conv4 = Conv2D(512, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(pool3)		# (batch, 64, 64, 512)
conv4 = Conv2D(512, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv4)		# (batch, 64, 64, 512)
drop4 = Dropout(0.5)(conv4)									# (batch, 64, 64, 512)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)				# (batch, 32, 32, 512)
conv5 = Conv2D(1024, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(pool4)		# (batch, 32, 32, 1024)
conv5 = Conv2D(1024, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv5)		# (batch, 32, 32, 1024)
drop5 = Dropout(0.5)(conv5)									# (batch, 32, 32, 1024)
up6 = UpSampling2D(size=(2, 2))(drop5)						# (batch, 64, 64, 1024)
up6 = Conv2D(512, 2, activation='relu', padding='same',
             kernel_initializer='he_normal')(up6)			# (batch, 64, 64, 512)
cat6 = Concatenate(axis=3)([drop4, up6])  # (batch, 64, 64, 1024)
conv6 = Conv2D(512, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(cat6)		# (batch, 64, 64, 512)
conv6 = Conv2D(512, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv6)		# (batch, 64, 64, 512)
up7 = UpSampling2D(size=(2, 2))(conv6)						# (batch, 128, 128, 512)
up7 = Conv2D(256, 2, activation='relu', padding='same',
             kernel_initializer='he_normal')(up7)			# (batch, 128, 128, 256)
cat7 = Concatenate(axis=3)([conv3, up7])				    # (batch, 128, 128, 512)
conv7 = Conv2D(256, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(cat7)		# (batch, 128, 128, 256)
conv7 = Conv2D(256, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv7)		# (batch, 128, 128, 256)
up8 = UpSampling2D(size=(2, 2))(conv7)						# (batch, 256, 256, 256)
up8 = Conv2D(128, 2, activation='relu', padding='same',
             kernel_initializer='he_normal')(up8)			# (batch, 256, 256, 128)
cat8 = Concatenate(axis=3)([conv2, up8])  					# (batch, 256, 256, 256)
conv8 = Conv2D(128, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(cat8)		# (batch, 256, 256, 128)
conv8 = Conv2D(128, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv8)		# (batch, 256, 256, 128)
up9 = UpSampling2D(size=(2, 2))(conv8)						# (batch, 512, 512, 128)
up9 = Conv2D(64, 2, activation='relu', padding='same',
             kernel_initializer='he_normal')(up9)			# (batch, 512, 512, 64)
cat9 = Concatenate(axis=3)([conv1, up9])  				    # (batch, 512, 512, 128)
conv9 = Conv2D(64, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(cat9)	 	# (batch, 512, 512, 64)
conv9 = Conv2D(64, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv9)		# (batch, 512, 512, 64)
conv9 = Conv2D(2, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv9)		# (batch, 512, 512, 2)
conv10 = Conv2D(1, 1, activation='sigmoid',
                name='conv10')(conv9)			            # (batch, 512, 512, 1)

model = Model(inputs=[inputs], outputs=[conv10])

# def train(self):
model.compile(optimizer=Adam(lr=1e-4),
              loss='binary_crossentropy', metrics=['accuracy'])


model_checkpoint = ModelCheckpoint(
	base_path + rel_result_path + 'unet.h5', monitor='loss', verbose=1, save_best_only=True)

model.fit({'inputs': im_train}, {'conv10': im_train_labels}, batch_size=4, epochs=10, verbose=1, callbacks=[
          model_checkpoint], validation_data=({'inputs': im_test}, {'conv10': im_test_labels}), shuffle=True)


# print('predict test data')
# im_test_labels_predict = model.predict(im_test, batch_size=1, verbose=1)
# np.save(base_path + rel_result_path + 'unet.h5', im_test_labels_predict)
# return im_test_labels_predict

# if __name__ == '__main__':
# unet_obj = cellcount_unet.get_unet()
# result = unet_obj.train()
