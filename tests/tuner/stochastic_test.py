#!/usr/bin/env python2
import os
import sys
import random
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from keras.datasets import mnist
from keras.utils import np_utils

import recognizer.tuner.stochastic as tuner_stochastic


# GLOBAL VARIABLES
TEST_DATA = "./train.csv"


class StochasticTests(unittest.TestCase):
    def test_random_walk(self):
        random.seed(1)

        # load data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
        X_train = X_train.astype("float32")
        X_test = X_test.astype("float32")
        X_train /= 255
        X_test /= 255

        # convert class vectors to binary class matrices
        nb_classes = 10
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        nn_data = {
            "X_train": X_train[:200],
            "Y_train": Y_train[:200],
            "X_test": X_test[:200],
            "Y_test": Y_test[:200],
            "batch_size": 20,
            "input_shape": (784,),
            "n_outputs": 10
        }

        tuner_stochastic.random_walk(
            nn_data=nn_data,
            chromo_size=400,
            max_iter=2,
            record_file_path="execution_test.dat",
            score_file_path="score_test.dat",
            error_file_path="error_test.dat"
        )


if __name__ == "__main__":
    unittest.main()
