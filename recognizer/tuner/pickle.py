#!/usr/bin/env python2
import sys
import cPickle


def create_pickle_index(path):
    index = open(path, "w")
    index.write("# chromosome, score, error, pickle file\n")
    return index


def save_pickle(index_file, path, obj, chromosome, score, err):
    sys.setrecursionlimit(40000)

    # pickle object to file
    f = file(path, 'wb')
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    # record in pickle index
    index_file.write("{0}, {1}, {2}, {3}\n".format(
        chromosome, score, err, path)
    )

def load_pickle(path):
    f = file(path, 'rb')
    loaded_obj = cPickle.load(f)
    f.close()

    return loaded_obj
