from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def c3d_model(summary=False):
    """ Return the Keras model of the network
    """
    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1), 
                            input_shape=(16,120,120,3)))
   # model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           border_mode='same', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='same', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='same', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='same', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv5a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv5b',
                            subsample=(1, 1, 1)))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool5'))
    model.add(BatchNormalization())
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096,activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(BatchNormalization())
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
   
    model.add(Dense(14, activation='softmax', name='fc9'))
    if summary:
        print(model.summary())
    return model

#model = get_model(summary=True)
