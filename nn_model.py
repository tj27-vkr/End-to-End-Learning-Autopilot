import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import losses, regularizers
from keras.layers.normalization import BatchNormalization


def base_model(img_height, img_width, img_channels):
    '''The proposed model.
    Input: img_height, img_width, img_channels'''
    model = Sequential()
    # Normalization
    model.add(Lambda(lambda x: x/255.,
                     input_shape=(img_height, img_width, img_channels)))
    # Cov layers
    model.add(Conv2D(24, kernel_size=(3, 3), padding='same',
                     activation='elu',
                     kernel_initializer='he_normal',
                     name='re_conv1'))
    model.add(BatchNormalization(name='re_bn1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='re_maxpool1'))

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
                    activation='elu',
                    kernel_initializer='he_normal',
                    name='re_conv2'))
    model.add(BatchNormalization(name='re_bn2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='re_maxpool2'))

    model.add(Conv2D(48, kernel_size=(3, 3), padding='same',
                    activation='elu',
                    kernel_initializer='he_normal',
                    name='re_conv3'))
    model.add(BatchNormalization(name='re_bn3'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='re_maxpool3'))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same',
                    activation='elu',
                    kernel_initializer='he_normal',
                    name='re_conv4'))
    model.add(BatchNormalization(name='re_bn4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='re_maxpool4'))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same',
                    activation='elu',
                    kernel_initializer='he_normal',
                    name='re_conv5'))
    model.add(BatchNormalization(name='re_bn5'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='re_maxpool5'))

    # Fullyconnected layer
    model.add(Flatten())
    model.add(Dense(1164,
                   activation='elu',
                   kernel_initializer='he_normal',
                   name='re_den1'))
    model.add(Dropout(0.2))

    model.add(Dense(100,
                   activation='elu',
                   kernel_initializer='he_normal',
                   name='re_den2'))
    model.add(Dropout(0.2))

    model.add(Dense(50,
                   activation='elu',
                   kernel_initializer='he_normal',
                   name='re_den3'))
    model.add(Dropout(0.5))

    model.add(Dense(10,
                   activation='elu',
                   kernel_initializer='he_normal',
                   name='re_den4'))
    model.add(Dropout(0.5))

    model.add(Dense(1, kernel_initializer='he_normal'))

    model.compile(loss='mse', optimizer="Adam")
    return model
