#!/usr/bin/python2
# import os
# import csv
# import copy
import math

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def plot_mnist_image(pixel_data, normalize=False):
    img = pixel_data.reshape((28, 28))

    # setup plot
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.cm.gray, interpolation="nearest")

    # Move left and bottom spines outward by 10 points
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    # hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # plot
    plt.show()


def plot_multiple_mnist_images(pixel_data, plot_format, normalize=False):
    x_plots = plot_format[0]
    y_plots = plot_format[1]
    plots = x_plots * y_plots

    # pre-check
    if pixel_data.shape[0] < plots:
        raise RuntimeError("Number of images exceed plot format!")
    elif math.fabs(pixel_data.shape[0] - plots) > x_plots:
        raise RuntimeError("Empty plot row detected!")

    # setup plot
    fig = plt.figure()

    # plot multiple images
    for i in range(plots):
        ax = fig.add_subplot(x_plots, y_plots, i + 1)
        img = pixel_data[i].reshape((96, 96))
        ax.imshow(img, cmap=plt.cm.gray, interpolation="nearest")

        # Move left and bottom spines outward by 10 points
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))

        # hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    # plot
    plt.show()


def load_mnist_data():
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

    return (X_train, Y_train, X_test, Y_test)


def read_csv(filename, delimiter=',', skiprows=1, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(",")
                for item in line:
                    yield dtype(item)
        read_csv.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, read_csv.rowlength))
    return data


# def save_to_csv(preds, file_path):
#     pd.DataFrame({
#         "ImageId": list(range(1, len(preds) + 1)),
#         "Label": preds}
#     ).to_csv(file_path, index=False, header=True)


def load_whale_data(train_file, test_file, dim=2):
    print("loading whale data")
    nb_classes=447

    # nomalize train data
    print("--> loading training data")
    train_data = read_csv(train_file)
    X_train = train_data[:, 1:]
    X_train = X_train.astype(np.float32)
    X_train = X_train / 255.0

    y_train = np.vstack(train_data[:, 0])
    y_train = y_train.astype(np.uint16)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    if dim == 2:
        X_train = X_train.reshape(-1, 1, 96, 96)
    Y_train = np_utils.to_categorical(y_train, 447)
    print("--> training data loaded")

    # nomalize test data
    print("--> loading test data")
    test_data = read_csv(test_file)
    X_test = test_data[:, 1:]
    X_test = X_test.astype(np.float32)
    X_test = X_test / 255.0

    y_test = np.vstack(test_data[:, 0])
    y_test = y_test.astype(np.uint16)

    X_test, y_test = shuffle(X_test, y_test, random_state=42)
    if dim == 2:
        X_test = X_test.reshape(-1, 1, 96, 96)
    Y_test = np_utils.to_categorical(y_test, 447)
    print("--> test data loaded")

    return (X_train, Y_train, X_test, Y_test)


def plot(fitlog):
    plt.plot(
        fitlog.epoch,
        fitlog.history['acc'],
        'g-',
        label="test accuracy"
    )
    plt.plot(
        fitlog.epoch,
        fitlog.history['val_acc'],
        'g--',
        label="validation accuracy"
    )
    plt.plot(
        fitlog.epoch,
        fitlog.history['loss'],
        'r-',
        label="test loss"
    )
    plt.plot(
        fitlog.epoch,
        fitlog.history['val_loss'],
        'r--',
        label="validation loss"
    )
    plt.legend()
    plt.show()
