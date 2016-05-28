#!/usr/bin/python2
import os
import sys
import errno
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from keras.datasets import mnist
from keras.utils import np_utils

import recognizer.mlp as mlp
import recognizer.data as data
import recognizer.tuner.mapping as mapping

# GLOBAL VARIABLES
TEST_DATA = "./train.csv"
WHALE_TRAIN_DATA = "/data/whale_data/first_attempt/train_data.csv"
WHALE_TEST_DATA = "/data/whale_data/first_attempt/test_data.csv"


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


class MLPTests(unittest.TestCase):
    # def test_analyze_mnist_data(self):
    #     (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #     X_train = X_train.reshape(60000, 784)
    #     X_test = X_test.reshape(10000, 784)
    #     X_train = X_train.astype("float32")
    #     X_test = X_test.astype("float32")
    #     X_train /= 255
    #     X_test /= 255
    #
    #     # convert class vectors to binary class matrices
    #     nb_classes = 10
    #     Y_train = np_utils.to_categorical(y_train, nb_classes)
    #     Y_test = np_utils.to_categorical(y_test, nb_classes)
    #
    #     results = mlp.analyze_mnist_data(
    #         x_train=X_train[:1000],
    #         y_train=Y_train[:1000],
    #         x_test=X_test[:1000],
    #         y_test=Y_test[:1000],
    #         batch_size=50,
    #         nb_epoch=10,
    #         result_file="test.dat"
    #     )


    # def test_mlp(self):
    #     X_train, Y_train, X_test, Y_test = data.load_whale_data(
    #         WHALE_TRAIN_DATA,
    #         WHALE_TEST_DATA,
    #         dim=1
    #     )
    #
    #     chromosome = [
    #         0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    #         0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    #         0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    #         0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    #         0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    #         0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    #         0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    #         0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    #     ]
    #     individual = {
    #         "chromosome": list(chromosome),
    #         "score": 0.0
    #     }
    #
    #     kwargs = mapping.keras_mapping_mlp(individual)
    #     kwargs["X_train"] = X_train
    #     kwargs["Y_train"] = Y_train
    #     kwargs["X_test"] = X_test
    #     kwargs["Y_test"] = Y_test
    #     kwargs["input_shape"] = (X_train.shape[1],)
    #     kwargs["nb_classes"] = 447
    #
    #     mlp.mlp(
    #         **kwargs
    #         # X_train=X_train,
    #         # y_train=Y_train,
    #         # X_test=X_test,
    #         # y_test=Y_test,
    #         #
    #         # input_shape=input_shape,
    #         # nb_classes=nb_classes,
    #         #
    #         # nb_layers=2,
    #         # hidden_neurons=[600, 400, 447],
    #         # activations=["tanh", "tanh", "tanh"],
    #         # dropouts=[0.25, 0.25, 0.25],
    #         # loss='categorical_crossentropy',
    #         # optimizer="adadelta",
    #         # nb_epoch=1,
    #         # batch_size=20,
    #     )

    def test_mlp_whale(self):
        X_train, Y_train, X_test, Y_test = data.load_whale_data(
            WHALE_TRAIN_DATA,
            WHALE_TEST_DATA,
            dim=1
        )

        input_shape = (X_train.shape[1],)
        nb_classes = 447

        for nb_neurons in [447, 500, 600, 800, 1000]:
            # for nb_neurons_2 in [447, 500, 600, 800, 1000]:
            #     for nb_neurons_3 in [447, 500, 600, 800, 1000]:
            #         target_path = "mlp_{0}_{1}_{2}_447".format(
            #             nb_neurons,
            #             nb_neurons_2,
            #             nb_neurons_3
            #         )
            target_path = "mlp_{0}_447".format(
                nb_neurons
            )
            mkdir_p(target_path)

            mlp.mlp(
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,

                input_shape=input_shape,
                nb_classes=nb_classes,

                nb_layers=2,
                hidden_neurons=[
                    nb_neurons,
                    447
                ],
                activations=["tanh", "tanh", "tanh", "tanh"],
                dropouts=[0.1, 0.1, 0.1, 0.1, 0.1],
                loss='categorical_crossentropy',
                optimizer="adadelta",
                nb_epoch=30,
                batch_size=100,

                model_file=os.path.join(target_path, "model.json"),
                results_file=os.path.join(target_path, "results.dat"),
                weights_file=os.path.join(target_path, "weights.dat")
            )


if __name__ == "__main__":
    unittest.main()
