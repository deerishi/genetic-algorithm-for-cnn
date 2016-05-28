#!/usr/bin/env python2
import os
import sys
import random
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import recognizer.data as data
import recognizer.tuner.ga as ga


# GLOBAL VARIABLE
WHALE_TRAIN_DATA = "/home/chutsu/whale_data/train.csv"
WHALE_TEST_DATA = "/home/chutsu/whale_data/test.csv"


if __name__ == "__main__":
    seed = int(sys.argv[1])
    random.seed(42 + seed)

    # neural network data
    X_train, Y_train, X_test, Y_test = data.load_whale_data(
        WHALE_TRAIN_DATA,
        WHALE_TEST_DATA
    )
    train_end = int(round(len(X_train) * 0.01))
    test_end = int(round(len(X_test) * 0.01))
    nn_data = {
        "X_train": X_train[:train_end],
        "Y_train": Y_train[:train_end],
        "X_test": X_test[:test_end],
        "Y_test": Y_test[:test_end],
        "input_shape": (1, 192, 192),
        "n_outputs": 447
    }

    # run
    ga.run(
        nn_data=nn_data,
        eval_func=ga.evaluate_cnn,
        chromo_size=500,
        max_gen=10,
        pop_size=10,
        t_size=2,
        record_file_path="exp{0}/execution_{0}.dat".format(seed),
        score_file_path="exp{0}/score_{0}.dat".format(seed),
        error_file_path="exp{0}/error_{0}.dat".format(seed),
        pickle_index_path="exp{0}/pickle.index".format(seed),
        pickle_path="exp{0}/".format(seed),
    )
