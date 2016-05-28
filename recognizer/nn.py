#!/usr/bin/env python
import math
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


def train_nn(x_train, y_train):
    print "training..."
    errors_record = []

    # intialize weights
    # uniform(a, b), b > a
    b = 1
    a = -1
    weights_dimension = (3, 1)
    weights_0 = (b - a) * np.random.random(weights_dimension) + a

    for i in xrange(1000):
        # forward propagation
        layer_0 = x_train
        layer_1 = sigmoid(np.dot(layer_0, weights_0))

        # delta rule - gradient descent
        total_error = y_train - layer_1
        layer_1_delta = total_error * sigmoid(layer_1, True)
        weights_0 += np.dot(layer_0.T, layer_1_delta)

        # record errors
        errors_record.append(np.absolute(total_error).sum())

    print "done training"
    print weights_0
    return weights_0, errors_record


def predict(x_test, y_test, weights_0):
    # predict
    correct = 0.0
    raw_error = 0.0
    data_size = len(x_test)

    for i in range(data_size):
        nn_output = sigmoid(np.dot(x_test[i], weights_0))

        # determine if classification is correct
        if y_test[i] == int(round(nn_output)):
            correct += 1

        # calculate raw error
        raw_error += math.fabs(y_test[i] - nn_output)

    # print summary
    percentage = ((correct / float(data_size)) * 100.0)
    print "raw error: %f" % (raw_error)
    print "classification percentage: %d%%" % percentage


if __name__ == "__main__":
    # seed random
    np.random.seed(1)

    # input data
    x_train = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    # output data
    y_train = np.array([
        [0, 0, 1, 1]
    ]).T

    weights_0, errors_record = train_nn(x_train, y_train)
    predict(x_train, y_train, weights_0)

    # plot errors
    plt.plot(errors_record)
    plt.xlabel("Epoch")
    plt.ylabel("Error (raw)")
    plt.show(block=True)
