#!/usr/bin/env python2
import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import recognizer.tuner.pickle as tuner_pickle


class DummyClass(object):
    def __init__(self):
        self.x = 1
        self.y = 1


class PickleTests(unittest.TestCase):
    def test_save_and_load_pickle(self):
        # setup
        index_file = tuner_pickle.create_pickle_index("index.pickle")
        obj = DummyClass()
        obj.x = 123
        obj.y = 456

        # save pickle
        tuner_pickle.save_pickle(index_file, "test.pickle", obj, "1001010101")

        # load pickle
        loaded_obj = tuner_pickle.load_pickle("test.pickle")

        # close index file
        index_file.close()
        os.remove("index.pickle")
        os.remove("test.pickle")

        # assert
        self.assertEquals(loaded_obj.x, obj.x)
        self.assertEquals(loaded_obj.y, obj.y)


if __name__ == "__main__":
    unittest.main()
