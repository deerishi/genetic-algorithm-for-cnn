#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

import recognizer.data as data


# MODEL PARAMETERS
batch_size = 128
nb_classes = 447
nb_epoch = 12
nb_pool = 2  # size of pooling area for max pooling

WHALE_TRAIN_DATA = "/home/chutsu/whale_data/train.csv"
WHALE_TEST_DATA = "/home/chutsu/whale_data/test.csv"


if __name__ == "__main__":
    # for reproducibility
    np.random.seed(1337)

    # load data
    X_train, Y_train, X_test, Y_test = data.load_whale_data(
        WHALE_TRAIN_DATA,
        WHALE_TEST_DATA
    )

    # CNN architecture
    model = Sequential()

    # layer 1
    model.add(Convolution2D(
        nb_filter=32,
        nb_row=11,
        nb_col=11,
        border_mode='valid',
        subsample=(4, 4),
        input_shape=(1, 192, 192))
    )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    # layer 2
    model.add(Convolution2D(
        nb_filter=64,
        nb_row=5,
        nb_col=5,
        border_mode='valid')
    )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    # layer 3
    model.add(Convolution2D(
        nb_filter=128,
        nb_row=3,
        nb_col=3,
        border_mode='valid')
    )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    # layer 4
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # layer 5
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # layer 6
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # compile loss and optimizer functions
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta'
    )

    # fit and evaluatte model
    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        show_accuracy=True,
        verbose=1,
        validation_data=(X_test, Y_test)
    )
    score = model.evaluate(
        X_test,
        Y_test,
        show_accuracy=True,
        verbose=0
    )

    print('Test score:', score[0])
    print('Test accuracy:', score[1])
