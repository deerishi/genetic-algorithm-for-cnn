#!/ussr/bin/env python
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import recognizer.data as data
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.utils import shuffle
from os.path import expanduser

home = expanduser("~")
print 'home is ',home
prefix=home+"/Kaggle data/"
WHALE_TRAIN_DATA = prefix+"96by 96 images red channel/train_data.csv"
WHALE_TEST_DATA = prefix+"96by 96 images red channel/test_data.csv"

Xtrain_file=prefix+'96by 96 images red channel/trainingData96_file'
Ytrain_file=prefix+'96by 96 images red channel/trainingLabel96_file'
Xtest_file=prefix+'96by 96 images red channel/testData96_file'
Ytest_file=prefix+'96by 96 images red channel/testLabel96_file'


imageWidth=96
imageHeight=96

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

def load_whale_data(train_file, test_file, nb_classes=447):
    print("loading whale data")

    # nomalize train data
    print("--> loading training data")
    train_data = read_csv(train_file)
    X_train = train_data[:, 1:]
    X_train = X_train.astype(np.float32)
    X_train = X_train / 255
    print 'here the shape is ',X_train.shape

    y_train = np.vstack(train_data[:, 0])
    y_train = y_train.astype(np.uint16)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
   #X_train = X_train.reshape(-1, 1, 192, 192)
    #Y_train = np_utils.to_categorical(y_train, 447)
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
    #X_test = X_test.reshape(-1, 1, 192, 192)
    #Y_test = np_utils.to_categorical(y_test, 447)
    print("--> test data loaded")

    return (X_train, y_train, X_test, y_test)
    
    



def loadData():
    X_train, Y_train, X_test, Y_test = load_whale_data(WHALE_TRAIN_DATA,WHALE_TEST_DATA)
    print 'shape os X_train is ',X_train.shape
    print 'shape os y_train is ',Y_train.shape
    print 'shape os X_test  is ',X_test.shape
    print 'shape os y_test is ',Y_test.shape
    np.save(Xtrain_file,X_train)
    np.save(Ytrain_file,Y_train)
    np.save(Xtest_file,X_test)
    np.save(Ytest_file,Y_test)
    
    
    #print 'data is ',X_train
    

if __name__=='__main__':
    loadData()
    

