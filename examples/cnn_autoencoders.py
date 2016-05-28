#!/usr/bin/env python2
import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from keras.models import Sequential
from keras.layers.core import AutoEncoder
from keras.layers.convolutional import Convolution2D

import recognizer.data as data


# GLOBAL VARIABLE
WHALE_TRAIN_DATA = "/data/whale_data/manual/train_data.csv"
WHALE_TEST_DATA = "/data/whale_data/manual/test_data.csv"


if __name__ == "__main__":
    # load data
    X_train, Y_train, X_test, Y_test = data.load_whale_data(
        WHALE_TRAIN_DATA,
        WHALE_TEST_DATA
    )

    model = Sequential()
    encoder = Convolution2D(
        100,
        32,
        32,
        input_shape=(1, 96, 96),
        border_mode="valid"
    )
    decoder = Convolution2D(
        1,
        96,
        96,
        input_shape=(100, 32, 32),
        border_mode="full"
    )
    model.add(
        AutoEncoder(
            encoder=encoder,
            decoder=decoder,
            output_reconstruction=False
        )
    )
    batch_size = 32
    nb_epoch = 10
    model.compile(loss='mse', optimizer='sgd')
    model.fit(
        X_train[0:100],
        X_train[0:100],
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        show_accuracy=True,
        validation_data=(X_test[0:100], X_test[0:100]),
        verbose=2
    )
