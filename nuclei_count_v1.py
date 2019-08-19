# In this version, the UNet is trained on 3 classes, and the network returns 
# probabilities for pixels to belong to each class (independently).
# Labels: 0 = background
# Labels: 1 = boundary
# Labels: 2 = foreground


import argparse
import os
import pdb
import time
import timeit
from random import shuffle

import numpy as np
import scipy.io as sio
import tensorflow as tf
from keras.callbacks import (Callback, CSVLogger,ModelCheckpoint,TensorBoard)
from keras.layers import (Activation, BatchNormalization,
                                            Concatenate, Conv2D, Cropping2D,
                                            Dense, Dropout, Input,
                                            MaxPooling2D, UpSampling2D)
from keras.models import Model
from keras.optimizers import Adam
from nuclei_count_data import DataClass, DataGenerator, IOpaths, validationData, trainingData

parser = argparse.ArgumentParser()
parser.add_argument("--run_mode",          default='train',       type=str,    help="Options; new, continue, infer")

parser.add_argument("--max_fg_frac",       default=0.8,           type=int,    help="Fraction of patches with foreground pixels in them")
parser.add_argument("--batch_size",        default=4,             type=int,    help="Batch size")
parser.add_argument("--n_steps_per_epoch", default=2000,          type=int,    help="Gradient steps per epoch")
parser.add_argument("--n_epoch",           default=1000,          type=int,    help="Number of epochs to train")
parser.add_argument("--warm_start",        default=0,             type=int,    help="Load weights")

parser.add_argument("--run_iter",          default=0,             type=int,    help="Run-specific id")
parser.add_argument("--model_id",          default='v1',          type=str,    help="Model name")
parser.add_argument("--exp_name",          default='nuclei_count',type=str,    help="Experiment name")


def main(run_mode='train',max_fg_frac=0.8,
         batch_size=4, n_steps_per_epoch=2000, n_epoch=1000, warm_start=0,
         run_iter=0, exp_name='nuclei_count', model_id='v1'):

    fileid = model_id + \
        '_fg_' + str(max_fg_frac) + \
        '_bs_' + str(batch_size) + \
        '_se_' + str(n_steps_per_epoch) + \
        '_ne_' + str(n_epoch) + \
        '_ri_' + str(run_iter)
    fileid = fileid.replace('.', '-')
    print(fileid)

    patch_size = 128
    chkpt_save_period = 100
    dir_pth = IOpaths(exp_name=exp_name)

    unet = Unet()
    unet.compile(optimizer=Adam(),loss={'output_im': loss_fcn_wbce})
    

    #Callbacks during training------------------------------------------------------
    csvlog = CSVLogger(filename=dir_pth['logs'] + fileid +'.csv')
    bestnet_cb = ModelCheckpoint(filepath=(dir_pth['result'] + fileid + '-bestwt.h5'),
                                 monitor='val_loss', verbose=1, save_best_only=True)
    history_cb = ModelCheckpoint(filepath=(dir_pth['checkpoints'] + fileid + '{epoch:04d}' + '.h5'),
                              verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=chkpt_save_period)

     #Training data-----------------------------------------------------------------
    train_fileid = ['268778_157',    '268778_77',     '271317_95',    '271317_143',
                    '275376_147',    '275376_175',    '275705_109',   '275705_131',
                    '279578_111',    '279578_123',    '299769_119',   '299769_167',
                    '299773_57',     '299773_68',     '301199_74',    '315576_116',
                    '316809_60',     '324471_113',    '330689_118',   '371808_52',
                    '386384_47',     '387573_103_1',  '389052_108',   '389052_119',
                    '352293-0020_1', '352293-0020_2', '370548-0034_1','370607-0034_1',
                    '370548-0034_2', '352293-0123',   '370607-0124_3','368669-0020', 
                    '370548-0034_3']                    
    
    training_dataObj_list = [DataClass(paths=dir_pth, file_id=f, pad=int(patch_size)) for f in train_fileid]

    #Validation data-----------------------------------------------------------------
    val_fileid = ['371808_60',   '387573_103', '324471_53', '333241_113',
                  '370607-0124', '370548-0034_4']

    valdata_fixed = validationData(file_ids=val_fileid, dir_pth=dir_pth, 
                                   max_fg_frac=max_fg_frac, patch_size=patch_size, n_patches_perfile=20)

    #Training Loop-------------------------------------------------------------------
    if warm_start==1:
        wt_file = dir_pth['result'] + 'warm-start-12-2018.h5'
        print('Loading weights from ' + wt_file)
        unet.load_weights(wt_file)
        traindata_fixed = trainingData(dataObj_list=training_dataObj_list, dir_pth=dir_pth, 
                                   max_fg_frac=max_fg_frac, patch_size=patch_size, n_patches_perfile=20)
        unet.fit(traindata_fixed[0]['input_im'],traindata_fixed[1]['output_im'],
                        validation_data=valdata_fixed,
                        initial_epoch=0, epochs=1, verbose=1,
                        callbacks=[history_cb, csvlog])
        unet.load_weights(wt_file)
        

    start_time = timeit.default_timer()
    for e in range(n_epoch):
        traindata_fixed = trainingData(dataObj_list=training_dataObj_list, dir_pth=dir_pth, 
                                   max_fg_frac=max_fg_frac, patch_size=patch_size, n_patches_perfile=20)

        #Rotate randomly for whole epoch data:
        rand_k=np.random.choice([0,1,2,3],size=1)[0]
        traindata_fixed[0]['input_im'] = np.rot90(traindata_fixed[0]['input_im'],k=rand_k,axes=(1,2))
        traindata_fixed[1]['output_im'] = np.rot90(traindata_fixed[1]['output_im'],k=rand_k,axes=(1,2))
        
        #Fit model here:
        _e=e+1
        unet.fit(traindata_fixed[0]['input_im'],traindata_fixed[1]['output_im'],
                        validation_data=valdata_fixed,
                        initial_epoch=e, epochs=_e, verbose=1,
                        callbacks=[history_cb, csvlog])
    elapsed = timeit.default_timer() - start_time
    print('Time elapsed: '+str(elapsed))
    return

def Unet():
    '''
    Return a Unet model. Hyperparameters appropriate set for the mesoscale connectivity dataset.
    '''
    conv_properties = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_normal'}
    input_im = Input(shape=(128, 128, 1), name='input_im')

    conv1 = Conv2D(filters=8, kernel_size=3, **conv_properties)(input_im)   # (batch,128, 128,   8)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)                           # (batch, 64,  64,   8)

    conv2 = Conv2D(filters=8, kernel_size=3, **conv_properties)(pool1)      # (batch, 64,  64,   8)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)                           # (batch, 32,  32,   8) 

    conv3 = Conv2D(filters=4, kernel_size=3, **conv_properties)(pool2)      # (batch, 32,  32,   4)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)                           # (batch, 16,  16,   4)
  
    conv4 = Conv2D(filters=4, kernel_size=3, **conv_properties)(pool3)      # (batch, 16,  16,   4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(rate=0.2)(conv4)                                        # (batch, 16,  16,   4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)                           # (batch,  8,   8,   4)

    conv5 = Conv2D(filters=4, kernel_size=3, **conv_properties)(pool4)      # (batch,  8,   8,   4)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(rate=0.2)(conv5)                                          # (batch,  8,   8,   4)

    up6   = UpSampling2D(size=(2, 2))(drop5)                                # (batch, 16,  16,   4)
    up6   = Conv2D(filters=4, kernel_size=2, **conv_properties)(up6)        # (batch, 16,  16,   4)
    cat6  = Concatenate(axis=3)([drop4, up6])                               # (batch, 16,  16,   4)
    conv6 = Conv2D(filters=4, kernel_size=3, **conv_properties)(cat6)       # (batch, 16,  16,   4)

    up7   = UpSampling2D(size=(2, 2))(conv6)                                # (batch, 32,  32,   4)
    up7   = Conv2D(4, 2, **conv_properties)(up7)                            # (batch, 32,  32,   4)
    cat7  = Concatenate(axis=3)([conv3, up7])                               # (batch, 32,  32,   8)
    conv7 = Conv2D(filters=4, kernel_size=3, **conv_properties)(cat7)       # (batch, 32,  32,   4)

    up8   = UpSampling2D(size=(2, 2))(conv7)                                # (batch, 64,  64,   4)
    up8   = Conv2D(filters=8, kernel_size=2, **conv_properties)(up8)        # (batch, 64,  64,   8)
    cat8  = Concatenate(axis=3)([conv2, up8])                               # (batch, 64,  64,  16)
    conv8 = Conv2D(filters=8, kernel_size=3, **conv_properties)(cat8)       # (batch, 64,  64,   8)

    up9   = UpSampling2D(size=(2, 2))(conv8)                                # (batch, 128, 128,  8)
    up9   = Conv2D(filters=8, kernel_size=2, **conv_properties)(up9)        # (batch, 128, 128,  8)
    cat9  = Concatenate(axis=3)([conv1, up9])                               # (batch, 128, 128, 16)
    conv9 = Conv2D(filters=8, kernel_size=3, **conv_properties)(cat9)       # (batch, 128, 128,  8) 

    output_im = Conv2D(filters=3,kernel_size=1, activation='softmax', name='output_im')(conv9)

    return Model(inputs=[input_im], outputs=[output_im])


def loss_fcn_wbce(y_true, y_pred):
    lbl  = y_true
    pred = y_pred
    weights = tf.stop_gradient(1. - tf.reduce_mean(lbl, axis=[0, 1, 2]))
    ce = - tf.multiply(lbl, tf.log(pred + tf.keras.backend.epsilon())) - \
           tf.multiply((1. - lbl), tf.log(1. - pred + tf.keras.backend.epsilon()))
    weighted_ce = tf.multiply(weights, ce)
    return tf.reduce_mean(weighted_ce, axis=None)
        
if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
