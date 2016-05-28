#!/usr/bin/python2
import os
import sys
import json
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import numpy as np
import matplotlib.pylab as plt

import recognizer.plot as plot


# GLOBAL VARIABLES
# GA_TEST_DATA = "./experiments/ga/exp8/execution_8.dat"
# GA_DATA_PATH = "./experiments/ga/"
GA_TEST_DATA_1 = "/data/ga_cnn_dataset_2/execution_test.dat"
GA_TEST_DATA_2 = "/data/ga_cnn_dataset_3/execution_test.dat"
RANDOM_WALK_DATA_PATH = "./experiments/random_walk/"
# RANDOM_WALK_SCORE_DATA = "./experiments/random_walk/"
# RESULTS_FILE = "./results.dat"
RESULTS_FILE = "/data/recognizer/cnn_relu/results.dat"
# RESULTS_PATH = "/data/models/mlp/2_layers"
RESULTS_PATH = "/data/recognizer/cnn_relu"
DATA_FILE = "./experiments/cnn/results(d2).dat"

class PlotTests(unittest.TestCase):
    # def test_load_csv(self):
    #     plot.load_csv(GA_TEST_DATA)

    def test_plot_results_all(self):
        path = "./experiments/cnn/results/d3/B.dat"
        plot.plot_results_file(path, True, "cnn_best.png")

        # path = "./experiments/cnn/results/d3"
        # plot.plot_results_all(path, True, "cnn_dataset3.png")
        #
        # path = "./experiments/cnn/results/d2"
        # plot.plot_results_all(path, True, "cnn_dataset2.png")

    # def test_plot_cnn(self):
    #     DATA_FILE = "./experiments/cnn/results.dat"
    #     plot.plot_results_file(DATA_FILE, True, "results.png")
    #
    #     DATA_FILE = "./experiments/cnn/results1.dat"
    #     plot.plot_results_file(DATA_FILE, True, "results1.png")
    #
    #     DATA_FILE = "./experiments/cnn/results2.dat"
    #     plot.plot_results_file(DATA_FILE, True, "results2.png")
    #
    #     DATA_FILE = "./experiments/cnn/results3.dat"
    #     plot.plot_results_file(DATA_FILE, True, "results3.png")
    #
    #     DATA_FILE = "./experiments/cnn/results4.dat"
    #     plot.plot_results_file(DATA_FILE, True, "results4.png")
    #
    #     DATA_FILE = "./experiments/cnn/results(d2).dat"
    #     plot.plot_results_file(DATA_FILE, True, "results(d2).png")
    #
    #     DATA_FILE = "./experiments/cnn/results1(d2).dat"
    #     plot.plot_results_file(DATA_FILE, True, "results1(d2).png")
    #
    #     DATA_FILE = "./experiments/cnn/results2(d2).dat"
    #     plot.plot_results_file(DATA_FILE, True, "results2(d2).png")


    # def test_plot_ga_cnn(self):
    #     data = {}
    #     data["dataset1"] = plot.load_csv(GA_TEST_DATA_1)
    #     data["dataset2"] = plot.load_csv(GA_TEST_DATA_2)
    #     plot.plot_ga_cnn(data, True, "./ga_cnn_tuner.pngear)

    # def test_plot_ga_runs(self):
    #     plot.plot_ga_runs(GA_DATA_PATH)

    # def test_plot_random_walk(self):
    #     data = plot.load_csv(TEST_DATA)
    #     plot.plot_random_walk(data)

    # def test_plot_random_walk_runs(self):
    #     plot.plot_random_walk_runs(RANDOM_WALK_DATA_PATH)

    # def test_plot_score_distribution(self):
    #     data = plot.load_score_csv(SCORE_DATA)
    #     plot.plot_score_distribution(data)

    # def test_load_all_score_csv(self):
    #     # data = plot.load_all_score_csv(RANDOM_WALK_DATA_PATH)
    #     data = plot.load_all_score_csv(GA_DATA_PATH)
    #     plot.plot_score_distribution(
    #         data,
    #         True,
    #         "test.png",
    #         "Score Distribution of Genetic Algorithm Tuning"
    #     )

    # def test_plot_results_file(self):
    #     plot.plot_results_file(RESULTS_FILE)

    # def test_plot_all_results(self):
    #     plot.plot_results(RESULTS_PATH)


if __name__ == "__main__":
    unittest.main()
