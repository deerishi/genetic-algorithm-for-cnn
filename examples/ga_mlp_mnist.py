#!/usr/bin/env python2
import os
import sys
import random
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from keras.datasets import mnist
from keras.utils import np_utils

import recognizer.tuner.ga as ga


if __name__ == "__main__":
    seed = int(sys.argv[1])
    random.seed(42 + seed)

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

    # neural network data
    train_end = int(round(len(X_train) * 0.1))
    test_end = int(round(len(X_test) * 0.1))
    nn_data = {
        "X_train": X_train[:train_end],
        "Y_train": Y_train[:train_end],
        "X_test": X_test[:test_end],
        "Y_test": Y_test[:test_end],
        "input_shape": (784,),
        "n_outputs": 10
    }

    # run
    ga.run(
        nn_data=nn_data,
        chromo_size=500,
        max_gen=30,
        pop_size=30,
        t_size=2,
        record_file_path="exp{0}/execution_{0}.dat".format(seed),
        score_file_path="exp{0}/score_{0}.dat".format(seed),
        error_file_path="exp{0}/error_{0}.dat".format(seed),
        pickle_index_path="exp{0}/pickle.index".format(seed),
        pickle_path="exp{0}/".format(seed),
    )
