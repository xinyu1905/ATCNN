# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dropout, Flatten, Dense, Activation


def ATCNN_Ef_model():

    input_shape = (10,10,1)
    model = Sequential()

    # layer1 
    model.add(Conv2D(8, kernel_size=(3, 3), padding='same',input_shape=input_shape))
    model.add(Activation('relu'))

    # layer2
    model.add(MaxPooling2D(pool_size=(2,2)))

    # layer3
    model.add(Flatten())
    model.add(Dense(50))
    model.add(Activation('relu'))

    # layer4
    model.add(Dense(30))
    model.add(Activation('relu'))

    # layer5
    model.add(Dense(1))
    model.add(Activation('linear'))
    

    model.compile(loss=keras.losses.mean_absolute_error,
            optimizer= keras.optimizers.Adadelta())
    return model

def ATCNN_Tc_model():

    input_shape = (10,10,1)
    model = Sequential()

    # layer1 
    model.add(Conv2D(64, kernel_size=(5, 5), padding='same',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # layer2
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # layer3
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # layer4
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # layer5
    model.add(Conv2D(64, kernel_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # layer6
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(200))

    # layer7
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    
    model.add(Dense(100))

    # layer8
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # layer9
    model.add(Dense(1))
    model.add(Activation('linear'))
    

    model.compile(loss=keras.losses.mean_absolute_error,
            optimizer= keras.optimizers.Adadelta())
    return model

