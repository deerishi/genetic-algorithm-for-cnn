#!/usr/bin/env python2
import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import recognizer.tuner.mapping as mapping


# GLOBAL VARIABLES
TEST_DATA = "./train.csv"


# class MappingTests(unittest.TestCase):
#     def test_binary2int(self):
#         retval = mapping.binary2int([0, 0, 0, 1])
#         self.assertEquals(retval, 1)
#
#         retval = mapping.binary2int([1, 0, 0, 1])
#         self.assertEquals(retval, 9)
#
#     def test_chromosome_pop_genes(self):
#         retval = mapping.chromosome_pop_genes([0, 1, 0, 0], 4)
#         self.assertEquals(retval, 4)
#
#     def test_activation_mapping(self):
#         retval = mapping.activation_mapping([0, 0, 0, 0])
#         self.assertEquals(retval, "model.add(Activation('sigmoid'))")
#
#     def test_dropout_mapping(self):
#         chromosome = [0, 0, 0, 0, 0, 1]
#         retval = mapping.dropout_mapping(chromosome)
#         self.assertEquals(retval, "model.add(Dropout(0.01))")
#
#     def test_sgd_optimizer(self):
#         retval = mapping.sgd_optimizer([0, 0, 0, 0, 0, 0, 1, 0])
#         self.assertEquals(
#             retval,
#             "optimizer = keras.optimizers.SGD(0.01, nesterov=False)"
#         )
#
#         retval = mapping.sgd_optimizer([1, 1, 1, 1, 1, 1, 1, 0])
#         self.assertEquals(
#             retval,
#             "optimizer = keras.optimizers.SGD(1.27, nesterov=False)"
#         )
#
#     def test_adagrad_optimizer(self):
#         retval = mapping.adagrad_optimizer([0, 0, 0, 0, 0, 0, 0, 0])
#         self.assertEquals(retval, "optimizer = keras.optimizers.Adagrad()")
#
#     def test_adadelta_optimizer(self):
#         retval = mapping.adadelta_optimizer([0, 0, 0, 0, 0, 0, 0, 0])
#         self.assertEquals(retval, "optimizer = keras.optimizers.Adadelta()")
#
#     def test_rmsprop_optimizer(self):
#         retval = mapping.rmsprop_optimizer([0, 0, 0, 0, 0, 0, 0, 0])
#         self.assertEquals(retval, "optimizer = keras.optimizers.RMSprop()")
#
#     def test_adam_optimizer(self):
#         retval = mapping.adam_optimizer([0, 0, 0, 0, 0, 0, 0, 0])
#         self.assertEquals(retval, "optimizer = keras.optimizers.Adam()")
#
#     def test_optimizer_mapping(self):
#         retval = mapping.optimizer_mapping([
#             0, 0,  # optimizer type
#             0, 0, 0, 0, 0, 0, 0,  # learning rate
#             0,  # nesterov
#             0, 0, 0  # error type
#         ])
#         expect = """\
# optimizer = keras.optimizers.SGD(0.0, nesterov=False)
# model.compile(loss='mse', optimizer=optimizer)
#         """.strip()
#
#         self.assertEquals(retval, expect)
#
#     def test_keras_mapping(self):
#         chromosome = [
#             0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
#             0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
#             0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
#             0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
#             0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
#         ]
#         individual = {
#             "chromosome": list(chromosome),
#             "score": 0.0
#         }
#         code = mapping.keras_mapping(
#             individual,
#             input_shape=(768,),
#             n_outputs=10
#         )
#         print code
#
#         self.assertEquals(individual["chromosome"], chromosome)
#         chromosome[0] = 1
#         self.assertNotEquals(individual["chromosome"], chromosome)
#         self.assertIsNotNone(code)

class MappingTests(unittest.TestCase):
    def test_keras_mapping2(self):
        chromosome = [
            0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
            0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
            0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        ]
        print len(chromosome)
        individual = {
            "chromosome": list(chromosome),
            "score": 0.0
        }
        m = mapping.keras_mapping2(individual)

        import pprint
        pprint.pprint(m)


if __name__ == "__main__":
    unittest.main()
