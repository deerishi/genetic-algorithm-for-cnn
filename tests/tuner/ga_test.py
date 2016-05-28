#!/usr/bin/env python2
import os
import sys
import random
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

# from keras.datasets import mnist
# from keras.utils import np_utils

import recognizer.data as data
import recognizer.tuner.ga as ga


# GLOBAL VARIABLES
TEST_DATA = "./train.csv"
WHALE_TRAIN_DATA = "/data/whale_data/manual/train_data.csv"
WHALE_TEST_DATA = "/data/whale_data/manual/test_data.csv"


class GaTests(unittest.TestCase):
    # def test_best_individual(self):
    #     # setup
    #     population = ga.init_population(pop_size=5)
    #     for individual in population:
    #         individual["score"] = random.random()
    #
    #     # find best individual
    #     best = ga.best_individual(population)
    #     population.remove(best)
    #
    #     # assert
    #     for i in population:
    #         self.assertTrue(best["score"] > i["score"])

    # def test_evaluate(self):
    #     # data
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
    #     nn_data = {
    #         "X_train": X_train,
    #         "Y_train": Y_train,
    #         "X_test": X_test,
    #         "Y_test": Y_test,
    #         "batch_size": 20,
    #         "input_shape": (784,),
    #         "n_outputs": 10
    #     }
    #
    #     # evaluate chromosome
    #     chromosome = [
    #         0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #         0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
    #     ]
    #     individual = {
    #         "chromosome": chromosome,
    #         "score": 0.0
    #     }
    #     pickle_index = pickle.create_pickle_index("pickle.index")
    #     eval_data = {
    #         "individual": individual,
    #         "nn_data": nn_data,
    #         "score_cache": {},
    #         "error_cache": {},
    #         "pickle_index": pickle_index,
    #         "pickle_path": "."
    #     }
    #     ga.evaluate(eval_data)
    #     ga.evaluate(eval_data)
    #
    #     # clean up
    #     os.remove("pickle.index")

    # def test_evaluate_population(self):
    #     population = ga.init_population(pop_size=10, chromo_size=200)
    #
    #     # load data
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
    #     nn_data = {
    #         "X_train": X_train[:50],
    #         "Y_train": Y_train[:50],
    #         "X_test": X_test[:50],
    #         "Y_test": Y_test[:50],
    #         "batch_size": 20,
    #         "input_shape": (784,),
    #         "n_outputs": 10
    #     }
    #     eval_data = {
    #         "nn_data": nn_data,
    #         "score_cache": {},
    #         "error_cache": {},
    #         "pickle_index": pickle.create_pickle_index("pickle.index"),
    #         "pickle_path": "."
    #     }
    #     population = ga.evaluate_population(
    #         population,
    #         eval_data
    #     )

    # def test_init_chromosome(self):
    #     chromosome = ga.init_chromosome()
    #     self.assertEquals(len(chromosome), 10)
    #     self.assertEquals(type(chromosome[0]), int)
    #
    # def test_init_population(self):
    #     population = ga.init_population()
    #     self.assertEquals(len(population), 100)
    #     self.assertEquals(type(population[0]["chromosome"][0]), int)
    #
    # def test_point_crossover(self):
    #     x1 = {"chromosome": [1, 2, 3, 4, 5], "score": None}
    #     x2 = {"chromosome": [6, 7, 8, 9, 10], "score": None}
    #     ga.point_crossover(1.0, x1, x2)
    #
    #     self.assertEquals(x1["chromosome"], [6, 7, 3, 4, 5])
    #     self.assertEquals(x2["chromosome"], [1, 2, 8, 9, 10])
    #
    # def test_point_mutation(self):
    #     x = {"chromosome": [1, 2, 3, 4, 5], "score": None}
    #     ga.point_mutation(1.0, x, 5)
    #     self.assertTrue(x["chromosome"] != [1, 2, 3, 4, 5])
    #
    # def test_tournament_selection(self):
    #     import random
    #     random.seed(1)
    #
    #     population = [
    #         {"chromosome": [1], "score": 1.0},
    #         {"chromosome": [2], "score": 0.0},
    #         {"chromosome": [3], "score": 0.0},
    #         {"chromosome": [4], "score": 2.0},
    #         {"chromosome": [5], "score": 0.0},
    #         {"chromosome": [6], "score": 0.0},
    #         {"chromosome": [7], "score": 5.0},
    #         {"chromosome": [8], "score": 0.0},
    #         {"chromosome": [9], "score": 0.0},
    #         {"chromosome": [10], "score": 1.0}
    #     ]
    #     new_population = ga.tournament_selection(population)
    #
    #     # import pprint
    #     # print ""
    #     # pprint.pprint(new_population)
    #     # print new_population[2]["chromosome"]
    #     # new_population[2]["chromosome"][0] = 2
    #     # print new_population[2]["chromosome"]
    #     # pprint.pprint(new_population)
    #
    #     self.assertEquals(len(new_population), 10)
    #     self.assertTrue(new_population != population)

    # def test_best_individual(self):
    #     population = [
    #         {"chromosome": [1], "score": 1.0},
    #         {"chromosome": [2], "score": 0.0},
    #         {"chromosome": [3], "score": 0.0},
    #         {"chromosome": [4], "score": 2.0},
    #         {"chromosome": [5], "score": 0.0},
    #         {"chromosome": [6], "score": 0.0},
    #         {"chromosome": [7], "score": 5.0},
    #         {"chromosome": [8], "score": 0.0},
    #         {"chromosome": [9], "score": 0.0},
    #         {"chromosome": [10], "score": 1.0}
    #     ]
    #
    #     best = ga.best_individual(population)
    #     self.assertEquals(best["score"], 5)

    def test_run(self):
        random.seed(42)
        X_train, Y_train, X_test, Y_test = data.load_whale_data(
            WHALE_TRAIN_DATA,
            WHALE_TEST_DATA
        )

        # neural network data
        nn_data = {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_test": X_test,
            "Y_test": Y_test,
            "input_shape": (1, 96, 96),
            "n_outputs": 447,
            "model_save_dir": "/data/nn_exp"
        }

        # run
        ga.run(
            nn_data=nn_data,
            chromo_size=500,
            max_gen=20,
            pop_size=20,
            t_size=2,
            record_file_path="execution_test.dat",
            score_file_path="score_test.dat",
            error_file_path="error_test.dat"
        )


if __name__ == "__main__":
    unittest.main()
