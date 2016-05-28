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
from os.path import expanduser


home = expanduser("~")






def analyze_whale_data(**kwargs):
    print("analyzing whale data")
    # seed random
    np.random.seed(kwargs["seed"])

    # model inputs and parameters
    #X_train = kwargs["X_train"]
    #X_test = kwargs["X_test"]
    #Y_train = kwargs["Y_train"]
    #Y_test = kwargs["Y_test"]
    
    X_train=np.load(home+'/gabor/numpyFiles/Training Set.npy')
    X_test=np.load(home+'/gabor/numpyFiles/TestSet.npy')
    Y_train=np.load(home+'/gabor/numpyFiles/Training Labels.npy')
    Y_test=np.load(home+'/gabor/numpyFiles/TestSet Labels.npy')
    
    X_test = X_test.reshape(-1, 1, 30, 96)
    Y_test = np_utils.to_categorical(Y_test, 447)
    
    
    X_train = X_train.reshape(-1, 1, 30, 96)
    Y_train = np_utils.to_categorical(Y_train, 447)
    
    img_rows = kwargs["img_rows"]
    img_cols = kwargs["img_cols"]

    nb_filters = kwargs.get("nb_filters", 32)
    nb_conv = kwargs.get("nb_conv", 3)
    nb_pool = kwargs.get("nb_pool", 2)

    batch_size = kwargs.get("batch_size", 32)
    nb_epoch = kwargs.get("nb_epoch", 12)
    nb_classes = kwargs.get("nb_classes", 447)

    print("X_test.shape == {};".format(X_test.shape))
    print("Y_test.shape == {};".format(Y_test.shape))
    print("X_test.shape == {};".format(X_train.shape))
    print("Y_test.shape == {};".format(Y_train.shape))
    
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
    model.add(Convolution2D(32, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    # layer 3
    model.add(Convolution2D(64, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    
    # layer 4
    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    
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
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
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
        for X_batch, Y_batch in datagen.flow(X_train, Y_train):
            loss, acc = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(X_batch.shape[0], values=[("train loss", loss), ("train accuracy", acc)])
        loss1[e] = loss
        acc1[e] = acc

        print("Testing...")
        # test time!
        progbar = generic_utils.Progbar(X_test.shape[0])
        for X_batch, Y_batch in datagen.flow(X_test, Y_test):
            score, test_acc = model.test_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(X_batch.shape[0], values=[("test loss", score), ("test accuracy", test_acc)])
        score1[e] = score
        test_acc1[e] = test_acc
    
    print("--> saving CNN")
    json_string = model.to_json()
    open('my_model_architecture.json', 'w').write(json_string)
    model.save_weights('my_model_weights.h5') 
    
    return (loss1, acc1, score1, test_acc1)
            




if __name__ == "__main__":
    
    
    
    loss1, acc1, score1, test_acc1 = analyze_whale_data(
        seed=42,

        

        img_rows=30,
        img_cols=96,

        nb_filters=32,

        nb_conv=3,
        nb_pool=2,

        batch_size=32,
        nb_epoch=300,
        nb_classes=447
    )
    
    #output_score = TemporaryFile()
    #np.save(output_score, score)
