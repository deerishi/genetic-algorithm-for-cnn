#!/usr/bin/env python
import os
import json

import numpy as np

from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

import pylab as pl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(
        data,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        cmap=cmap
    )
    pl.colorbar(im, cax=cax)


def analyze_mnist_data(**kwargs):
    X_train = kwargs["X_train"]
    y_train = kwargs["y_train"]
    X_test = kwargs["X_test"]
    y_test = kwargs["y_test"]
    batch_size = kwargs["batch_size"]

    nb_filters = 32
    nb_conv = 3
    nb_pool = 2
    nb_classes = 10
    img_rows = 28
    img_cols = 28
    nb_epoch = 12

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # CNN architecture
    model = Sequential()

    # layer 1
    model.add(
        Convolution2D(
            nb_filters,
            nb_conv,
            nb_conv,
            border_mode='full',
            input_shape=(1, img_rows, img_cols)
        )
    )
    model.add(Activation('relu'))

    # layer 2
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    # layer 3
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # layer 4
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        show_accuracy=True,
        verbose=1,
        validation_data=(X_test, Y_test)
    )
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def train_whale_data(**kwargs):
    print("analyzing whale data")
    # seed random
    np.random.seed(kwargs["seed"])

    # model inputs and parameters
    data_augmentation = kwargs.get("data_augmentation", False)
    X_train = kwargs["X_train"]
    X_test = kwargs["X_test"]
    Y_train = kwargs["Y_train"]
    Y_test = kwargs["Y_test"]

    img_rows = kwargs["img_rows"]
    img_cols = kwargs["img_cols"]

    nb_filters = kwargs.get("nb_filters", 32)
    nb_conv = kwargs.get("nb_conv", 3)
    nb_pool = kwargs.get("nb_pool", 2)

    batch_size = kwargs["batch_size"]
    nb_epoch = kwargs.get("nb_epoch", 12)
    nb_classes = kwargs.get("nb_classes", 10)

    model_file = kwargs["model_file"]
    weights_file = kwargs["weights_file"]
    results_file = kwargs["results_file"]

    # CNN architecture
    print("--> creating CNN network ...")
    results = {
        "acc": [],
        "val_acc": [],
        "loss": [],
        "val_loss": []
    }
    model = Sequential()

    # layer 1
    model.add(
        Convolution2D(
            nb_filters,
            nb_conv,
            nb_conv,
            input_shape=(1, img_rows, img_cols),
        )
    )
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # layer 2
    model.add(
        Convolution2D(
            nb_filters,
            nb_conv,
            nb_conv,
            input_shape=(1, img_rows, img_cols),
        )
    )
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # layer 3
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.4))

    # layer 4
    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # compile, fit and evaluate model
    print("--> compiling CNN functions")
    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd'
    )

    # fit model
    print("--> fitting CNN")
    if data_augmentation is False:
        print "--> fitting data"
        fitlog = model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            show_accuracy=True,
            verbose=1,
            validation_data=(X_test, Y_test)
        )
        results = fitlog.history

    else:
        # turn on data augmentation
        print "--> augmenting data"
        datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
        )
        datagen.fit(X_train)

        print "--> fitting data"
        for e in range(nb_epoch):
            print "epoch:", e
            for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size):
                train_loss, train_accuracy = model.train_on_batch(
                    X_batch,
                    Y_batch,
                    accuracy=True
                )
                valid_loss, valid_accuracy = model.test_on_batch(
                    X_test,
                    Y_test,
                    accuracy=True
                )

            results["acc"].append(float(train_accuracy))
            results["val_acc"].append(float(valid_accuracy))
            results["loss"].append(float(train_loss))
            results["val_loss"].append(float(valid_loss))

            print "acc: {0}".format(train_accuracy),
            print "val_acc: {0}".format(valid_accuracy),
            print "acc_loss: {0}".format(train_loss),
            print "val_loss: {0}".format(valid_loss)

    # save model
    model_data = model.to_json()
    model_file = open(model_file, "w")
    model_file.write(json.dumps(model_data))
    model_file.close()

    # save model weights
    model.save_weights(weights_file, overwrite=True)

    # save results
    results["nb_epoch"] = nb_epoch
    results["batch_size"] = batch_size
    rf = open(results_file, "w")
    rf.write(json.dumps(results))
    rf.close()


def evaluate_model(X_test, Y_test, model_file, weights_file):
    model = None
    loss = None
    accuracy = None

    # load NN
    if os.path.isfile(weights_file):
        print("--> loading NN model")
        # load model data
        model_file = open(model_file, "r")
        model_data = json.loads(model_file.read())
        model_file.close()
        model = model_from_json(model_data)

        # load model weights
        model.load_weights(weights_file)
    else:
        raise RuntimeError("failed to load [{0}]".format(weights_file))

    # evaluate model
    if model:
        print("--> compiling CNN")
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adadelta'
        )

        print("--> evaluating CNN")
        score = model.evaluate(
            X_test,
            Y_test,
            show_accuracy=True,
            verbose=0
        )
        loss = score[0]
        accuracy = score[1]

        # print score and accuracy
        print("Test score:", score[0])
        print("Test accuracy:", score[1])

    return loss, accuracy



def cnn(**kwargs):
    verbose = kwargs.get("verbose", True)
    X_train = kwargs["X_train"]
    Y_train = kwargs["Y_train"]
    X_test = kwargs["X_test"]
    Y_test = kwargs["Y_test"]
    input_shape = kwargs["input_shape"]
    nb_classes = kwargs["nb_classes"]
    data_augmentation = kwargs.get("data_augmentation", True)

    nb_convo_layers = kwargs["nb_convo_layers"]
    nb_filters = kwargs["nb_filters"]
    nb_conv = kwargs["nb_conv"]

    convo_activations = kwargs["convo_activations"]
    maxpools = kwargs["maxpools"]
    pool_sizes = kwargs["pool_sizes"]
    convo_dropouts = kwargs["convo_dropouts"]

    nb_dense_layers = kwargs["nb_dense_layers"]
    dense_hidden_neurons = kwargs["dense_hidden_neurons"]
    dense_activations = kwargs["dense_activations"]
    dense_dropouts = kwargs["dense_dropouts"]

    loss = kwargs["loss"]
    optimizer = kwargs["optimizer"]
    nb_epoch = kwargs["nb_epoch"]
    batch_size = kwargs["batch_size"]

    model_file = kwargs.get("model_file")
    weights_file = kwargs.get("weights_file")
    results_file = kwargs.get("results_file")
    results = {
     "acc": [],
     "loss": [],
     "val_acc": [],
     "val_loss": []
    }

    # CNN architecture
    model = Sequential()

    # convolution layers
    for i in range(nb_convo_layers):
        # convo layer
        if i == 0:
            model.add(
                Convolution2D(
                    nb_filters[i],
                    nb_conv[i],
                    nb_conv[i],
                    input_shape=input_shape
                )
            )
        else:
            model.add(
                Convolution2D(
                    nb_filters[i],
                    nb_conv[i],
                    nb_conv[i],
                    border_mode='valid',
                )
            )

        # activation
        if convo_activations[i]:
            model.add(Activation(convo_activations[i]))

        # max-pooling
        if maxpools[i]:
            model.add(MaxPooling2D(pool_size=(pool_sizes[i], pool_sizes[i])))

        # dropout
        if convo_dropouts[i]:
            model.add(Dropout(convo_dropouts[i]))

    # dense layers
    model.add(Flatten())
    for i in range(nb_dense_layers):
        # dense layer
        if (i + 1) == nb_dense_layers:
            model.add(Dense(nb_classes))
        else:
            model.add(Dense(dense_hidden_neurons[i]))

        # activation
        if dense_activations[i]:
            model.add(Activation(dense_activations[i]))

        # dropout
        if dense_dropouts[i]:
            model.add(Dropout(dense_dropouts[i]))

    # loss function and optimizer
    if verbose:
        print("--> compiling CNN")
    model.compile(loss=loss, optimizer=optimizer)

    # fit model
    if verbose:
        print("--> fitting CNN")

    if data_augmentation is False:
        fitlog = model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            show_accuracy=True,
            verbose=verbose,
            validation_data=(X_test, Y_test)
        )
        results = fitlog.history

    else:
        # turn on data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=True,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True
        )
        datagen.fit(X_train)

        for e in range(nb_epoch):
            if verbose:
                print "epoch:", e

            tmp_train_acc = []
            tmp_train_loss = []
            tmp_test_acc = []
            tmp_test_loss = []
            train_batch_counter = 0
            test_batch_counter = 0

            for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size):
                train_loss, train_accuracy = model.train_on_batch(
                    X_batch,
                    Y_batch,
                    accuracy=True
                )

                tmp_train_acc.append(train_accuracy)
                tmp_train_loss.append(train_loss)
                train_batch_counter += 1

            for X_batch, Y_batch in datagen.flow(X_test, Y_test, batch_size):
                valid_loss, valid_accuracy = model.test_on_batch(
                    X_batch,
                    Y_batch,
                    accuracy=True
                )
                tmp_test_acc.append(valid_accuracy)
                tmp_test_loss.append(valid_loss)
                test_batch_counter += 1

            epoch_train_acc = sum(tmp_train_acc) / float(train_batch_counter)
            epoch_train_loss = sum(tmp_train_loss) / float(train_batch_counter)
            epoch_test_acc = sum(tmp_test_acc) / float(test_batch_counter)
            epoch_test_loss = sum(tmp_test_loss) / float(test_batch_counter)

            results["acc"].append(epoch_train_acc)
            results["loss"].append(epoch_train_loss)
            results["val_acc"].append(epoch_test_acc)
            results["val_loss"].append(epoch_test_loss)

            if verbose:
                print "acc: {0}".format(epoch_train_acc),
                print "loss: {0}".format(epoch_train_loss),
                print "val_acc: {0}".format(epoch_test_acc),
                print "val_loss: {0}".format(epoch_test_loss)

    # save model
    if model_file:
        model_data = model.to_json()
        model_file = open(model_file, "w")
        model_file.write(json.dumps(model_data))
        model_file.close()

    # save model weights
    if weights_file:
        model.save_weights(weights_file, overwrite=True)

    # save results
    if results_file:
        results["nb_epoch"] = nb_epoch
        results["batch_size"] = batch_size
        rf = open(results_file, "w")
        rf.write(json.dumps(results))
        rf.close()

    # evaluate
    score = model.evaluate(
        X_test,
        Y_test,
        show_accuracy=True,
        verbose=verbose,
        batch_size=batch_size
    )

    return results, score


def load_cnn(dataset, model_file, weights_file):
    model_file = open(model_file, "r")
    model_json = json.loads(model_file.read())
    weights_file = open(weights_file, "r")

    model = model_from_json(model_json)
    model.load_weights(weights_file)

    X_train = dataset["X_train"]
    Y_train = dataset["Y_train"]
    X_test = dataset["X_test"]
    Y_test = dataset["Y_test"]

    model.predict(X_train)
