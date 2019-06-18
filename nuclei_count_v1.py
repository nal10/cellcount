# In this version, the UNet is trained on 3 classes, and the network returns probabilities of a given pixel belonging to each class. 
# Labels: 0 = background
# Labels: 1 = boundary
# Labels: 2 = foreground


import pdb
import os
import time
import timeit
from random import shuffle

import numpy as np
import scipy.io as sio
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, Callback, CSVLogger
from keras.layers import (Concatenate, Conv2D, Cropping2D, Dropout, Input,
                          MaxPooling2D, UpSampling2D, Dense, Activation)
from keras.models import Model
from keras.optimizers import Adam

from nuclei_count_data import IOpaths,DataClass,DataGenerator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_mode",        default='train',      type=str,    help="Options; new, continue, infer")
parser.add_argument("--batch_size",      default=4,            type=int,    help="Batch size")
parser.add_argument("--steps_per_epoch", default=200,          type=int,    help="Gradient steps per epoch")
parser.add_argument("--n_epoch",         default=1000,         type=int,    help="Number of epochs to train")
parser.add_argument("--warm_start",      default=0,            type=int,    help="Load weights")

parser.add_argument("--run_iter",        default=0,            type=int,    help="Run-specific id")
parser.add_argument("--model_id",        default='v1',         type=str,    help="Model name")
parser.add_argument("--exp_name",        default='nuclei_count',type=str,   help="Experiment name")


def main(run_mode='train',
         batch_size=4, steps_per_epoch=200, n_epoch=1000, warm_start=0,
         run_iter=0, exp_name='nuclei_count', model_id='v1'):

    fileid = model_id + \
        '_bs_' + str(batch_size) + \
        '_se_' + str(steps_per_epoch) + \
        '_ne_' + str(n_epoch) + \
        '_ri_' + str(run_iter)
    fileid = fileid.replace('.', '-')
    print(fileid)

    patch_size=128
    save_period=int(1e2)

    dir_pth = IOpaths(exp_name=exp_name)
    
    #Training data
    train_fileid = ['268778_157',    '268778_77',     '271317_95',    '271317_143',
                    '275376_147',    '275376_175',    '275705_109',   '275705_131',
                    '279578_111',    '279578_123',    '299769_119',   '299769_167',
                    '299773_57',     '299773_68',     '301199_74',    '315576_116',
                    '316809_60',     '324471_113',    '330689_118',   '371808_52',
                    '386384_47',     '387573_103_1',  '389052_108',   '389052_119',
                    '352293-0020_1', '352293-0020_2', '370548-0034_1','370607-0034_1',
                    '370548-0034_2', '352293-0123']

    val_fileid = ['371808_60', '387573_103', '324471_53', '333241_113',
                '368669-0020', '370607-0124']
                        
    train_generator = DataGenerator(file_ids=train_fileid, dir_pth=dir_pth, max_fg_frac=0.5, batch_size=4, patch_size=128 ,n_steps_per_epoch=200)
    unet = UNET()

    def loss_fcn_wbce(y_true, y_pred):
        lbl = y_true
        pred = y_pred
        weights = tf.stop_gradient(1. - tf.reduce_mean(lbl, axis=[0, 1, 2]))
        ce = - tf.multiply(lbl, tf.log(pred + tf.keras.backend.epsilon())) - \
            tf.multiply((1. - lbl), tf.log(1. - pred + tf.keras.backend.epsilon()))
        weighted_ce = tf.multiply(weights, ce)
        return tf.reduce_mean(weighted_ce, axis=None)

    unet.compile(optimizer=Adam(),loss={'output_im': loss_fcn_wbce})

    bestnet_cb = ModelCheckpoint(filepath=(dir_pth['result'] + fileid + '-bestwt.h5'),
                                monitor='loss', verbose=1, save_best_only=True)

    hist_cb = ModelCheckpoint(filepath=(dir_pth['checkpoints']+ fileid + '/' + '{epoch:04d}' + '.h5'),
                            verbose=1, save_best_only=False, save_weights_only=True,
                            mode='auto', period=save_period)

    csvlog = CSVLogger(filename=dir_pth['logs'] + model_id +'.csv')

    if warm_start==1:
        print('Loading weights ...')
        unet.load_weights(dir_pth['result'] + 'warm-start-12-2018.h5')
        print('Done')

    start_time = timeit.default_timer()
    unet.fit_generator(train_generator,
                       initial_epoch=0, epochs=n_epoch, verbose=1,
                       callbacks=[bestnet_cb, hist_cb, csvlog])
    elapsed = timeit.default_timer() - start_time
    print('Time elapsed: '+str(elapsed))
    return

def UNET():
    #Define Unet model. Model hyperparameters were tuned on the mesoscale dataset.
    conv_properties = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_normal'}
    input_im = Input(shape=(128, 128, 1), name='input_im')

    conv1 = Conv2D(filters=8, kernel_size=3, **conv_properties)(input_im)   # (, 128,128,8)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)	                        # (, 64, 64, 8)

    conv2 = Conv2D(filters=8, kernel_size=3, **conv_properties)(pool1)      # (, 64, 64, 8)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)	                        # (, 32, 32, 8)

    conv3 = Conv2D(filters=4, kernel_size=3, **conv_properties)(pool2)		# (, 32, 32, 4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)	                        # (, 16, 16, 4)

    conv4 = Conv2D(filters=4, kernel_size=3, **conv_properties)(pool3)		# (, 16, 16, 4)
    drop4 = Dropout(rate=0.2)(conv4)						                # (, 16, 16, 4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)	                        # (, 8, 8, 4)

    conv5 = Conv2D(filters=4, kernel_size=3, **conv_properties)(pool4)		# (, 8, 8, 4)
    drop5 = Dropout(rate=0.2)(conv5)						                # (, 8, 8, 4)
    
    up6 =   UpSampling2D(size=(2, 2))(drop5)			                    # (, 16, 16, 4)
    up6 =   Conv2D(filters=4, kernel_size=2, **conv_properties)(up6)		# (, 16, 16, 4)
    cat6 =  Concatenate(axis=3)([drop4, up6])  		                        # (, 16, 16, 4)
    conv6 = Conv2D(filters=4, kernel_size=3, **conv_properties)(cat6)		# (, 16, 16, 4)

    up7 =   UpSampling2D(size=(2, 2))(conv6)			                    # (, 32, 32, 4)
    up7 =   Conv2D(4, 2, **conv_properties)(up7)		                    # (, 32, 32, 4)
    cat7 =  Concatenate(axis=3)([conv3, up7])		                        # (, 32, 32, 8)
    conv7 = Conv2D(filters=4, kernel_size=3, **conv_properties)(cat7)		# (, 32, 32, 4)

    up8 =   UpSampling2D(size=(2, 2))(conv7)			                    # (, 64, 64, 4)
    up8 =   Conv2D(filters=8, kernel_size=2, **conv_properties)(up8)		# (, 64, 64, 8)
    cat8 =  Concatenate(axis=3)([conv2, up8])  		                        # (, 64, 64, 16)
    conv8 = Conv2D(filters=8, kernel_size=3, **conv_properties)(cat8)		# (, 64, 64, 8)

    up9 = UpSampling2D(size=(2, 2))(conv8)			                        # (, 128,128, 8)
    up9 = Conv2D(filters=8, kernel_size=2, **conv_properties)(up9)		    # (, 128,128, 8)
    cat9 = Concatenate(axis=3)([conv1, up9])  		                        # (, 128,128, 16)
    conv9 = Conv2D(filters=8, kernel_size=3, **conv_properties)(cat9)		# (, 128,128, 8) 
    output_im = Conv2D(filters=3,kernel_size=1, activation='softmax', name='output_im')(conv9)

    model = Model(inputs=[input_im], outputs=[output_im])
    return model


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))


