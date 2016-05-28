#!/usr/bin/python2
import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import numpy as np

import matplotlib.cm as cm
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers.core import Dense
from keras.layers.core import AutoEncoder

import recognizer.data as data

# GLOBAL VARIABLES
WHALE_TRAIN_DATA = "/data/whale_data/manual_rotated/train_data_gray_rotate.csv"
WHALE_TEST_DATA = "/data/whale_data/manual_rotated/test_data_gray_rotate.csv"


def plot(weights):

    print len(weights[0][0]), weights[0]

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        img = np.array(weights[0][i]).reshape(24, 24)
        plt.imshow(img, cmap=cm.Greys_r)
    plt.show()


class AutoEncoderTests(unittest.TestCase):
    # def test_analyze_mnist_data(self):
    #     batch_size = 32
    #     nb_epoch = 1
    #     nb_classes = 10
    #
    #     # load data
    #     X_train, Y_train, X_test, Y_test = data.load_mnist_data()
    #
    #     # layer-wise pretraining
    #     encoders = []
    #     # nb_hidden_layers = [784, 600, 500, 400]
    #     # nb_hidden_layers = [784, 576, 400]
    #     nb_hidden_layers = [784]
    #     X_train_tmp = np.copy(X_train[:1000])
    #
    #     for i in range(len(nb_hidden_layers) - 1):
    #         n_in, n_out = nb_hidden_layers[i: i + 2]
    #         print "training layer with input: {0} -> output: {1}".format(
    #             n_in,
    #             n_out
    #         )
    #
    #         # create and train
    #         ae = Sequential()
    #         ae.add(
    #             AutoEncoder(
    #                 encoder=Dense(n_out, input_dim=n_in, activation='sigmoid'),
    #                 decoder=Dense(n_in, input_dim=n_out, activation='sigmoid'),
    #                 output_reconstruction=False,
    #             )
    #         )
    #
    #         # compile
    #         ae.compile(loss="mean_squared_error", optimizer="rmsprop")
    #
    #         # fit
    #         ae.fit(
    #             X_train_tmp,
    #             X_train_tmp,
    #             batch_size=batch_size,
    #             nb_epoch=nb_epoch
    #         )
    #
    #         # store trainined weight and update training data
    #         encoders.append(ae.layers[0].encoder)
    #         X_train_tmp = ae.predict(X_train_tmp)
    #
    #     # put it together
    #     model = Sequential()
    #     for encoder in encoders:
    #         model.add(encoder)
    #     model.add(Dense(nb_classes, activation='softmax'))
    #
    #     model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    #     model.fit(
    #         X_train,
    #         Y_train,
    #         batch_size=batch_size,
    #         nb_epoch=nb_epoch,
    #         show_accuracy=True,
    #         verbose=1,
    #         validation_data=(X_test, Y_test)
    #     )
    #     score = model.evaluate(
    #         X_test,
    #         Y_test,
    #         show_accuracy=True,
    #         verbose=1
    #     )
    #
    #     print('Test score:', score[0])
    #     print('Test accuracy:', score[1])

    # def test_denoising_autoencoder(self):
    #     batch_size = 32
    #     nb_epoch = 20
    #     nb_classes = 10
    #
    #     # load data
    #     X_train, Y_train, X_test, Y_test = data.load_mnist_data()
    #
    #     # layer-wise pretraining
    #     X_train_tmp = np.copy(X_train)
    #     nb_neurons = 400
    #
    #     # create and train
    #     ae = Sequential()
    #     ae.add(
    #         AutoEncoder(
    #             encoder=Dense(nb_neurons, input_dim=784, activation='sigmoid'),
    #             decoder=Dense(784, input_dim=nb_neurons, activation='sigmoid'),
    #             output_reconstruction=True,
    #         )
    #     )
    #
    #     # compile
    #     ae.compile(loss="mean_squared_error", optimizer="rmsprop")
    #
    #     # fit
    #     ae.fit(
    #         X_train_tmp,
    #         X_train_tmp,
    #         batch_size=batch_size,
    #         nb_epoch=nb_epoch
    #     )
    #     results = ae.predict(X_train, verbose=1)
    #
    #     img_data = np.concatenate((X_train[0:9], results[0:9]))
    #     data.plot_multiple_mnist_images(img_data, (6, 3))

    def test_denoising_autoencoder(self):
        # # load data
        X_train, Y_train, X_test, Y_test = data.load_whale_data(
            WHALE_TRAIN_DATA,
            WHALE_TEST_DATA,
            dim=1
        )

        # autoencoder parameters
        batch_size = 32
        nb_epoch = 1000
        nb_neurons = 600
        img_dim = X_train.shape[1]

        # create and train
        X_train_tmp = np.copy(X_train)
        ae = Sequential()
        ae.add(
            AutoEncoder(
                encoder=Dense(nb_neurons, input_dim=img_dim, activation='sigmoid'),
                decoder=Dense(img_dim, input_dim=nb_neurons, activation='sigmoid'),
                output_reconstruction=True,
            )
        )

        # compile
        ae.compile(loss="mean_squared_error", optimizer="adadelta")

        # fit
        ae.fit(
            X_train_tmp,
            X_train_tmp,
            batch_size=batch_size,
            nb_epoch=nb_epoch
        )
        results = ae.predict(X_train, verbose=1)
        np.save("results_2.npy", results)

        img_data = np.concatenate((X_train[0:9], results[0:9]))
        data.plot_multiple_mnist_images(img_data[0:9], (6, 3))

        # results = np.load("results.npy")
        # data.plot_multiple_mnist_images(X_train[0:9], (3, 3))
        # data.plot_multiple_mnist_images(results[0:9], (3, 3))



if __name__ == "__main__":
    unittest.main()
