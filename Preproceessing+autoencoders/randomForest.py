#!/usr/bin/env python
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
from keras.layers.core import AutoEncoder
from keras.layers import containers
from keras.optimizers import RMSprop,SGD
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt



Xtrain_file='/home/drishi/64 by 64 proper head images/projectFiles/trainingData64_file.npy'
Ytrain_file='/home/drishi/64 by 64 proper head images/projectFiles/trainingLabel64_file.npy'
Xtest_file='/home/drishi/64 by 64 proper head images/projectFiles/testData64_file.npy'
Ytest_file='/home/drishi/64 by 64 proper head images/projectFiles/testLabel64_file.npy'

covxFile='/home/drishi/64 by 64 proper head images/projectFiles/CocX64.npy'
UFromSVD_file='/home/drishi/64 by 64 proper head images/projectFiles/UfromSVD64_file.npy'
VFromSVD_file='/home/drishi/64 by 64 proper head images/projectFiles/VfromSVD64_file.npy'
SFromSVD_file='/home/drishi/64 by 64 proper head images/projectFiles/SfromSVD64_file.npy'

WhitenedData64='/home/drishi/64 by 64 proper head images/projectFiles/FullDatasetNoDimReducedWhitenedData64.npy'
WhitenedTestData64='/home/drishi/64 by 64 proper head images/projectFiles/FullDatasetNoDimReducedWhitenedTestData64.npy'


def classify():
    X_train=np.load(WhitenedData64)
    Y_train=np.load(Ytrain_file)
    X_test=np.load(WhitenedTestData64)
    Y_test=np.load(Ytest_file)  
    print 'shape os X_train is ',X_train.shape
    print 'shape os y_train is ',Y_train.shape
    print 'shape os X_test  is ',X_test.shape
    print 'shape os y_test is ',Y_test.shape
    
    #First Step of Whitening is Centering the datasets
    rf=RandomForestClassifier(n_estimators=200)
    rf.fit(X_train,Y_train.ravel())
    predictions=rf.predict(X_test)
    gold=Y_test.ravel()
    
    counter=0
    for i in range(0,len(predictions)):
        if(gold[i]==predictions[i]):
            counter=counter+1
    print 'res is ', counter
    
    savetxt('/home/drishi/64 by 64 proper head images/projectFiles/predict1.csv', rf.predict(X_test), delimiter=',', fmt='%f')
    
   
if __name__=='__main__':
    classify()        
    

