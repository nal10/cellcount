# Adapted from https://github.com/zhixuhao/unet

import os

import numpy as np
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import (Conv2D, Cropping2D, Dropout, Input, MaxPooling2D,
                          UpSampling2D, merge)
from keras.models import Model
from keras.optimizers import Adam


class cellcount_unet(object):

	def __init__(self):
		pass

	def load_data(self):
		# Replace with fileIO functions
		im_train = []
		im_train_labels = []
		im_test = []
		im_test_labels = []
		return im_train, im_train_labels, im_test, im_test_labels

	def get_unet(self):
		from keras.models import Model
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
		drop4 = Dropout(0.5)(conv4)									# (batch, 64, 64, 512) - DROPOUT!
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)				# (batch, 32, 32, 512)
		conv5 = Conv2D(1024, 3, activation='relu', padding='same',
		               kernel_initializer='he_normal')(pool4)		# (batch, 32, 32, 1024)
		conv5 = Conv2D(1024, 3, activation='relu', padding='same',
		               kernel_initializer='he_normal')(conv5)		# (batch, 32, 32, 1024)
		drop5 = Dropout(0.5)(conv5)									# (batch, 32, 32, 1024) - DROPOUT!
		up6 = UpSampling2D(size=(2, 2))(drop5)						# (batch, 64, 64, 1024)
		up6 = Conv2D(512, 2, activation='relu', padding='same',
		             kernel_initializer='he_normal')(up6)			# (batch, 64, 64, 512)
		merge6 = merge([drop4, up6], mode='concat',
		               concat_axis=3)  # (batch, 64, 64, 1024)
		conv6 = Conv2D(512, 3, activation='relu', padding='same',
		               kernel_initializer='he_normal')(merge6)		# (batch, 64, 64, 512)
		conv6 = Conv2D(512, 3, activation='relu', padding='same',
		               kernel_initializer='he_normal')(conv6)		# (batch, 64, 64, 512)
		up7 = UpSampling2D(size=(2, 2))(conv6)						# (batch, 128, 128, 512)
		up7 = Conv2D(256, 2, activation='relu', padding='same',
		             kernel_initializer='he_normal')(up7)			# (batch, 128, 128, 256)
		merge7 = merge([conv3, up7], mode='concat',
		               concat_axis=3)  # (batch, 128, 128, 512)
		conv7 = Conv2D(256, 3, activation='relu', padding='same',
		               kernel_initializer='he_normal')(merge7)		# (batch, 128, 128, 256)
		conv7 = Conv2D(256, 3, activation='relu', padding='same',
		               kernel_initializer='he_normal')(conv7)		# (batch, 128, 128, 256)
		up8 = UpSampling2D(size=(2, 2))(conv7)						# (batch, 256, 256, 256)
		up8 = Conv2D(128, 2, activation='relu', padding='same',
		             kernel_initializer='he_normal')(up8)			# (batch, 256, 256, 128)
		merge8 = merge([conv2, up8], mode='concat',
		               concat_axis=3)  # (batch, 256, 256, 256)
		conv8 = Conv2D(128, 3, activation='relu', padding='same',
		               kernel_initializer='he_normal')(merge8)		# (batch, 256, 256, 128)
		conv8 = Conv2D(128, 3, activation='relu', padding='same',
		               kernel_initializer='he_normal')(conv8)		# (batch, 256, 256, 128)
		up9 = UpSampling2D(size=(2, 2))(conv8)						# (batch, 512, 512, 128)
		up9 = Conv2D(64, 2, activation='relu', padding='same',
		             kernel_initializer='he_normal')()				# (batch, 512, 512, 64)
		merge9 = merge([conv1, up9], mode='concat',
		               concat_axis=3) 	# (batch, 512, 512, 128)
		conv9 = Conv2D(64, 3, activation='relu', padding='same',
		               kernel_initializer='he_normal')(merge9)	 	# (batch, 512, 512, 64)
		conv9 = Conv2D(64, 3, activation='relu', padding='same',
		               kernel_initializer='he_normal')(conv9)		# (batch, 512, 512, 64)
		conv9 = Conv2D(2, 3, activation='relu', padding='same',
		               kernel_initializer='he_normal')(conv9)		# (batch, 512, 512, 2)
		conv10 = Conv2D(1, 1, activation='sigmoid', name='conv10')(
			conv9)			# (batch, 512, 512, 1)

		model = Model(input=[inputs], output=[conv10])

		model.compile(optimizer=Adam(lr=1e-4),
		              loss='binary_crossentropy', metrics=['accuracy'])

		return model

	def train(self):

		print("loading data")
		im_train, im_train_labels, im_test, _ = self.load_data()
		print("loading data done")
		model = self.get_unet()
		print("got unet")

		model_checkpoint = ModelCheckpoint(
			'unet.h5', monitor='loss', verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(im_train, im_train_labels, batch_size=4, nb_epoch=10, verbose=1,
		          validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

		print('predict test data')

		im_test_labels_predict = model.predict(im_test, batch_size=1, verbose=1)
		#np.save('../dat/results/predicted_labels.npy', im_test_labels_predict)
		return im_test_labels_predict


if __name__ == '__main__':
	unet_obj = cellcount_unet.get_unet()
	result = unet_obj.train()
