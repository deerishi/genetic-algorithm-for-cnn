#!/usr/bin/env python2
import os
import sys
import random
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import recognizer.data as data
import recognizer.tuner.ga as ga

# GLOBAL VARIABLES
WHALE_TRAIN_DATA = "/data/whale_data/manual_rotated/train_data.csv"
WHALE_TEST_DATA = "/data/whale_data/manual_rotated/test_data.csv"


if __name__ == "__main__":
    random.seed(42)
    X_train, Y_train, X_test, Y_test = data.load_whale_data(
        WHALE_TRAIN_DATA,
        WHALE_TEST_DATA
    )

    # neural network data
    train_end = len(X_train) / 0.5
    test_end = len(X_test) / 0.5
    nn_data = {
        "X_train": X_train[0: train_end],
        "Y_train": Y_train[0: train_end],
        "X_test": X_test[0: test_end],
        "Y_test": Y_test[0: test_end],
        "input_shape": (1, 96, 96),
        "n_outputs": 447,
        "model_save_dir": "/data/ga_cnn_dataset_3"
    }

    # run
    ga.run(
        nn_data=nn_data,
        chromo_size=500,
        max_gen=10,
        pop_size=10,
        t_size=2,
        record_file_path="/data/ga_cnn_dataset_3/execution_test.dat",
        score_file_path="/data/ga_cnn_dataset_3/score_test.dat",
        error_file_path="/data/ga_cnn_dataset_3/error_test.dat"
    )
