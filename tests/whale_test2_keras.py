#!/usr/bin/python2
import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import math

from tempfile import TemporaryFile
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils, generic_utils
from keras.preprocessing.image import ImageDataGenerator


# GLOBAL VARIABLE
WHALE_TRAIN_DATA = "train.csv"
WHALE_TEST_DATA = "test.csv"

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


def analyze_whale_data(**kwargs):
    print("analyzing whale data")
    # seed random
    np.random.seed(kwargs["seed"])

    # model inputs and parameters
    X_train = kwargs["X_train"]
    X_test = kwargs["X_test"]
    Y_train = kwargs["Y_train"]
    Y_test = kwargs["Y_test"]

    img_rows = kwargs["img_rows"]
    img_cols = kwargs["img_cols"]

    nb_filters = kwargs.get("nb_filters", 32)
    nb_conv = kwargs.get("nb_conv", 3)
    nb_pool = kwargs.get("nb_pool", 2)

    batch_size = kwargs.get("batch_size", 32)
    nb_epoch = kwargs.get("nb_epoch", 12)
    nb_classes = kwargs.get("nb_classes", 447)

    # CNN architecture
    print("--> creating CNN network ...")
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
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    # layer 3
<<<<<<< HEAD
    '''model.add(Convolution2D(64, 5, 5, border_mode='valid'))
=======
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
>>>>>>> 83679dd23777dfb3a495d9944aed2f21abefb5ce
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    
    # layer 4
    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))'''
    
    # layer 5
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # layer 6
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # layer 4
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # compile, fit and evaluate model
    print("--> compiling CNN functions")
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta'
    )
    print("--> fitting CNN")
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=True,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images
    
    datagen.fit(X_train)
    
    loss1 = np.zeros((nb_epoch))
    loss1 = loss1.astype(np.float32)
    acc1 = np.zeros((nb_epoch))
    acc1 = acc1.astype(np.float32)
    score1 = np.zeros((nb_epoch))
    score1 = score1.astype(np.float32)
    test_acc1 = np.zeros((nb_epoch))
    test_acc1 = test_acc1.astype(np.float32)    
    
    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")
        # batch train with realtime data augmentation
        progbar = generic_utils.Progbar(X_train.shape[0])
        temp1 = 0.0
        temp2 = 0.0
        for X_batch, Y_batch in datagen.flow(X_train, Y_train):
            loss, acc = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(X_batch.shape[0], values=[("train loss", loss), ("train accuracy", acc)])
            temp1 = temp1 + loss
            temp2 = temp2 + acc
        loss1[e] = temp1/115
        acc1[e] = temp2/115
        print(acc1[e])

        print("Testing...")
        # test time!
        progbar = generic_utils.Progbar(X_test.shape[0])
        temp1 = 0.0
        temp2 = 0.0
        for X_batch, Y_batch in datagen.flow(X_test, Y_test):
            score, test_acc = model.test_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(X_batch.shape[0], values=[("test loss", score), ("test accuracy", test_acc)])
            temp1 = temp1 + score
            temp2 = temp2 + test_acc
        score1[e] = temp1/29
        test_acc1[e] = temp2/29
        print(test_acc1[e])
    
    print("--> saving CNN")
    json_string = model.to_json()
    open('my_model_architecture.json', 'w').write(json_string)
    model.save_weights('my_model_weights.h5') 
    
    return (loss1, acc1, score1, test_acc1)
            

def load_whale_data(train_file, test_file, nb_classes=447):
    print("loading whale data")

    # nomalize train data
    print("--> loading training data")
    train_data = read_csv(train_file)
    X_train = train_data[:, 1:]
    X_train = X_train.astype(np.float32)
    X_train = X_train / 255

    y_train = np.vstack(train_data[:, 0])
    y_train = y_train.astype(np.uint16)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_train = X_train.reshape(-1, 1, 96, 96)
    Y_train = np_utils.to_categorical(y_train, 447)
    print("--> training data loaded")

    # nomalize test data
    print("--> loading test data")
    test_data = read_csv(test_file)
    X_test = test_data[:, 1:]
    X_test = X_test.astype(np.float32)
    X_test = X_test / 255

    y_test = np.vstack(test_data[:, 0])
    y_test = y_test.astype(np.uint16)

    X_test, y_test = shuffle(X_test, y_test, random_state=42)
    X_test = X_test.reshape(-1, 1, 96, 96)
    Y_test = np_utils.to_categorical(y_test, 447)
    print("--> test data loaded")

    return (X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    
    X_train, Y_train, X_test, Y_test = load_whale_data(WHALE_TRAIN_DATA, WHALE_TEST_DATA)
    print("X_test.shape == {};".format(X_test.shape))
    print("Y_test.shape == {};".format(Y_test.shape))
    
    figs, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i in range(4):
        for j in range(4):
            axes[i, j].imshow(-X_test[i + 4 * j].reshape(96, 96), cmap='gray', interpolation='none')
            axes[i, j].set_xticks([])
            axes[i, j].axis('off')
    
    loss1, acc1, score1, test_acc1 = analyze_whale_data(
        seed=42,

        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,

        img_rows=96,
        img_cols=96,

        nb_filters=32,
        nb_conv=3,
        nb_pool=2,

        batch_size=32,
        nb_epoch=200,
        nb_classes=447
    )
    np.save("train_loss.npy", loss1)
    np.save("train_acc.npy", acc1)
    np.save("test_loss.npy", score1)
    np.save("test_acc.npy", test_acc1)
