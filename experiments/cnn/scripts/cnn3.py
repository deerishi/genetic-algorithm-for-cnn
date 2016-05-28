#!/usr/bin/env python2
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import recognizer.cnn as cnn
import recognizer.data as data

# GLOBAL VARIABLE
WHALE_TRAIN_DATA = "../train.csv"
WHALE_TEST_DATA = "../test.csv"


if __name__ == "__main__":
    # load data
    X_train, Y_train, X_test, Y_test = data.load_whale_data(
        WHALE_TRAIN_DATA,
        WHALE_TEST_DATA
    )

    print("X_test.shape == {};".format(X_test.shape))
    print("Y_test.shape == {};".format(Y_test.shape))

    # configuration
    kwargs = {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test,
        "input_shape": (1, 96, 96),
        "nb_classes": 447,
        "data_augmentation": True,

        "nb_convo_layers": 5,
        "nb_filters": [32, 64, 64, 128, 128],
        "nb_conv": [3, 3, 3, 3, 3],

        "convo_activations": ["relu", "relu", "relu", "relu", "relu"],
        "maxpools": [True, False, True, False, True],
        "pool_sizes": [2, 2, 2, 2, 2],
        "convo_dropouts": [None, None, None, None, None],

        "nb_dense_layers": 3,
        "dense_hidden_neurons": [1000, 1000, 447],
        "dense_activations": ["relu", "relu", "softmax"],
        "dense_dropouts": [0.5, 0.5, None],

        "loss": "categorical_crossentropy",
        "optimizer": "adadelta",
        "nb_epoch": 200,
        "batch_size": 32,

        "model_file": "model3.json",
        "weights_file": "weights3.dat",
        "results_file": "results3.dat"
    }

    # run cnn
    cnn.cnn(**kwargs)
