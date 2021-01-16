from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.layers import Dense,Flatten, Conv3D, MaxPool3D,Input,BatchNormalization , Dropout, Activation
from keras.optimizers import Adadelta
from keras.optimizers import SGD,adam, Adagrad
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.models import model_from_json
import numpy as np
import cv2
import os
from math import floor
import tensorflow as tf

def c3d_model():
    classes = 2
    
    """ Return the Keras model of the network
    """
    model = Sequential()
    input_shape=(16, 100, 100, 3) # l, h, w, c
   
    model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1',
                            input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3a'))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4a'))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5a'))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5b'))
    
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    #model.add(Dense(2048, activation='relu', name='fc6'))
    #model.add(Dropout(.5))
    #model.add(Dense(2048, activation='relu', name='fc7'))
    #model.add(Dropout(.5))
    #model.add(Dense(2, activation='sigmoid', name='fc8'))

    
    #WS FC Layer with custom parameters
    model.add(Dense(512,input_dim=4608,W_regularizer=l2(0.001),activation='relu', name='fc6'))
    model.add(Dropout(.6))
    model.add(Dense(32,activation='relu',W_regularizer=l2(0.001), name='fc7'))
    model.add(Dropout(.6))
    model.add(Dense(2, activation='sigmoid',W_regularizer=l2(0.001), name='fc8'))
    return model