#!/usr/bin/python2
import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import recognizer.data as data


# GLOBAL VARIABLE
WHALE_TRAIN_DATA = "/home/chutsu/whale_data/train.csv"
WHALE_TEST_DATA = "/home/chutsu/whale_data/test.csv"


class DataTests(unittest.TestCase):
    def test_load_mnist_data(self):
        X_train, Y_train, X_test, Y_test = data.load_mnist_data()
        self.assertEquals(X_train.shape, (60000, 784))
        self.assertEquals(X_test.shape, (10000, 784))
        self.assertEquals(Y_train.shape, (60000, 10))
        self.assertEquals(Y_test.shape, (10000, 10))

    def test_read_csv(self):
        retval = data.read_csv(WHALE_TRAIN_DATA, 1)
        print retval.shape

    def test_whale_data(self):
        X_train, Y_train, X_test, Y_test = data.load_whale_data(
            WHALE_TRAIN_DATA,
            WHALE_TEST_DATA
        )


if __name__ == "__main__":
    unittest.main()
