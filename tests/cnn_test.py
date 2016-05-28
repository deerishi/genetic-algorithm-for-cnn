#!/usr/bin/python2
import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import recognizer.data as data
import recognizer.cnn as cnn


# GLOBAL VARIABLE
WHALE_TRAIN_DATA = "/data/whale_data/manual/train_data.csv"
WHALE_TEST_DATA = "/data/whale_data/manual/test_data.csv"


class CNNTests(unittest.TestCase):
    # def test_analyze_mnist_data(self):
    #     X_train, Y_train, X_test, Y_test = data.load_mnist_data()
    #     cnn.analyze_mnist_data(
    #         X_train=X_train,
    #         Y_test=Y_test,
    #         X_test=X_test,
    #         Y_test=Y_test,
    #         batch_size=50
    #     )

    # def test_whale_data(self):
    #     X_train, Y_train, X_test, Y_test = data.load_whale_data(
    #         WHALE_TRAIN_DATA,
    #         WHALE_TEST_DATA
    #     )
    #     cnn.train_whale_data(
    #         seed=42,
    #
    #         X_train=X_train,
    #         X_test=X_test,
    #         Y_train=Y_train,
    #         Y_test=Y_test,
    #
    #         data_augmentation=True,
    #
    #         img_rows=96,
    #         img_cols=96,
    #
    #         nb_filters=32,
    #         nb_conv=3,
    #         nb_pool=3,
    #
    #         batch_size=32,
    #         nb_epoch=40,
    #         nb_classes=447,
    #
    #         model_file="model.dat",
    #         weights_file="weights.dat",
    #         results_file="results.dat"
    #     )

    # def test_whale_data(self):
    #     X_train, Y_train, X_test, Y_test = data.load_whale_data(
    #         WHALE_TRAIN_DATA,
    #         WHALE_TEST_DATA
    #     )
    #     cnn.train_whale_data(
    #         seed=42,
    #
    #         X_train=X_train,
    #         X_test=X_test,
    #         Y_train=Y_train,
    #         Y_test=Y_test,
    #
    #         data_augmentation=True,
    #
    #         img_rows=96,
    #         img_cols=96,
    #
    #         nb_filters=20,
    #         nb_conv=3,
    #         nb_pool=3,
    #
    #         batch_size=32,
    #         nb_epoch=40,
    #         nb_classes=447,
    #
    #         model_file="model.dat",
    #         weights_file="weights.dat",
    #         results_file="results.dat"
    #     )

    # def test_evaluate_model(self):
    #     X_train, Y_train, X_test, Y_test = data.load_whale_data(
    #         WHALE_TRAIN_DATA,
    #         WHALE_TEST_DATA
    #     )
    #     cnn.evaluate_model(X_test, Y_test, "model.json", "weights.dat")

    # def test_cnn(self):
    #     img_rows = 28
    #     img_cols = 28
    #     X_train, Y_train, X_test, Y_test = data.load_mnist_data()
    #     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    #     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    #
    #     kwargs = {
    #         "X_train": X_train[:100],
    #         "Y_train": Y_train[:100],
    #         "X_test": X_test[:100],
    #         "Y_test": Y_test[:100],
    #         "input_shape": (1, 28, 28),
    #         "nb_classes": 10,
    #
    #         "nb_convo_layers": 2,
    #         "nb_filters": [32, 32],
    #         "nb_conv": [3, 3],
    #
    #         "convo_activations": ["relu", "relu"],
    #         "maxpools": [False, True],
    #         "pool_sizes": [None, 2],
    #         "convo_dropouts": [None, 0.25],
    #
    #         "nb_dense_layers": 1,
    #         "dense_hidden_neurons": [10],
    #         "dense_activations": ["relu"],
    #         "dense_dropouts": [0.5],
    #
    #         "loss": "categorical_crossentropy",
    #         "optimizer": "adadelta",
    #         "nb_epoch": 12,
    #         "batch_size": 50,
    #
    #         "weights_file": "weights.dat",
    #         "results_file": "results.dat"
    #     }
    #     fitlog, score = cnn.cnn(**kwargs)

    def test_cnn_whale(self):
        X_train, Y_train, X_test, Y_test = data.load_whale_data(
            WHALE_TRAIN_DATA,
            WHALE_TEST_DATA
        )
        # X_train = []
        # Y_train = []
        # X_test = []
        # Y_test = []

        kwargs = {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_test": X_test,
            "Y_test": Y_test,
            "input_shape": (1, 192, 192),
            "nb_classes": 477,

            "nb_convo_layers": 2,
            "nb_filters": [32, 32],
            "nb_conv": [3, 3],

            "convo_activations": ["relu", "relu"],
            "maxpools": [False, True],
            "pool_sizes": [None, 2],
            "convo_dropouts": [None, 0.25],

            "nb_dense_layers": 1,
            "dense_hidden_neurons": [477],
            "dense_activations": ["relu"],
            "dense_dropouts": [0.5],

            "loss": "categorical_crossentropy",
            "optimizer": "adadelta",
            "nb_epoch": 30,
            "batch_size": 32,

            "weights_file": "weights.hdf5",
            "results_file": "results.dat"
        }
        fitlog, score = cnn.cnn(**kwargs)

if __name__ == "__main__":
    unittest.main()
