#!/usr/bin/env python2
import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


import recognizer.data as data
import recognizer.logreg as logreg


# GLOBAL VARIABLES
TEST_DATA = "./train.csv"


class LogisticRegressionTests(unittest.TestCase):

    def test_train(self):
        images, labels = data.load_mnist_data(TEST_DATA)

        # create train / test data
        dataset = data.random_split_mnist_data(images, labels, 0.6)
        imgs_train, imgs_test, labels_train, labels_test = dataset

        # from test split test and validate data
        dataset = data.random_split_mnist_data(imgs_test, labels_test, 0.5)
        imgs_test, imgs_validate, labels_test, labels_validate = dataset

        # dataset
        dataset = {
            "train": (imgs_train, labels_train),
            "test": (imgs_test, labels_test),
            "validate": (imgs_validate, labels_validate)
        }

        # run logistic regression
        logreg.sgd_mnist(dataset)


if __name__ == "__main__":
    unittest.main()
