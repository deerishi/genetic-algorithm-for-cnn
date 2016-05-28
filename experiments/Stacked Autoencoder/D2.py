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
from keras.preprocessing.image import ImageDataGenerator
import json

WHALE_TRAIN_DATA = "/home/drishi/64 by 64 proper head images/train_data.csv"
WHALE_TEST_DATA = "/home/drishi/64 by 64 proper head images/test_data.csv"

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


prefix='/home/drishi/model Files From AutoEncoder Exp/'

batch_size = 69
nb_classes = 447
nb_epoch = 300

f=open('/home/drishi/64 by 64 proper head images/ab.txt','a')

XwhiteTrain=np.load(Xtrain_file)
Y_train=np.load(Ytrain_file)
def train():

    model=Sequential()

    print 'X.shape is ',XwhiteTrain.shape
    print 'Y.shape is ',Y_train.shape
    nb_hidden_layers = [len(XwhiteTrain[0]),700,500,300]
    XtrainKeras=XwhiteTrain.reshape(-1,len(XwhiteTrain[0]))
    pretraining_files=['Layer_1_PCAWhitened.txt','Layer_1_PCAWhitened.txt','Layer_3_PCAWhitened.txt']
    print 'shape of XTrain Keras is ',XtrainKeras.shape
    YtrainKeras=np_utils.to_categorical(Y_train, nb_classes)
    op1=RMSprop(lr=0.01,rho=0.8,epsilon=1e-3)
    
    #datagen=ImageDataGenerator(zca_whitening=True)
    #datagen.fit()
    
    X_train_tmp=XtrainKeras
    trained_encoders=[]
    

    #XtrainKeras=XwhiteTrain.reshape(-1,1,len(XwhiteTrain),len(XwhiteTrain[0]))
    #YtrainKeras=np_utils.to_categorical(Y_train, nb_classes)

    
    #XtestKeras=X_test.reshape(-1,1,imageWidth,imageHeight)
    #YtestKeras=np_utils.to_categorical(Y_test, nb_classes)
    #X_train_tmp=XtrainKeras
    i=0
    for n_in, n_out in zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]):
        print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
        # Create AE and training
        ae = Sequential()
        encoder = containers.Sequential([Dense( n_out,input_dim=n_in, activation='sigmoid')])
        decoder = containers.Sequential([Dense( n_in,input_dim=n_out, activation='sigmoid')])
        ae.add(AutoEncoder(encoder=encoder, decoder=decoder,output_reconstruction=False))
        ae.compile(loss='mean_squared_error', optimizer=op1)
        hist=ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=nb_epoch)
        #f.write(str(hist.history))
        Fname=prefix+'autoencoder_n_in='+str(n_in)+'_n_out= '+str(n_out)+'.json'
        weightName=prefix+'Weights_autoencoder_n_in='+str(n_in)+'_n_out= '+str(n_out)+'.h5'
        json_string = model.to_json()
        f2=open(pretraining_files[i],'wb')
        f2.write(json.dumps(hist.history)) 
        f2.close()
        i=i+1
        open(Fname, 'w').write(json_string) 
        model.save_weights(weightName,overwrite=True)
        # Store trainined weight
        trained_encoders.append(ae.layers[0].encoder)
        # Update training data
        X_train_tmp = ae.predict(X_train_tmp)
        f.close()
        
        
    #ae1=Sequential()
    #encoder1=containers.Sequential([Dense(len(XwhiteTrain[0])-200,len(XwhiteTrain[0]),activation='sigmoid')])    
    X_test=np.load(Xtest_file)
    Y_test=np.load(Ytest_file)
    Y_test=np_utils.to_categorical(Y_test, nb_classes)
    X_test=X_test.reshape(-1,len(X_test[0]))
    print 'shape of X_test  is ',X_test.shape
    print('Fine-tuning')
    sgd=SGD(lr=0.01, momentum=0.5, decay=0., nesterov=False)

    i=1
    model = Sequential()
    for encoder in trained_encoders:
        model.add(encoder)
    model.add(Dense( nb_classes,input_dim=nb_hidden_layers[-1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    hist=model.fit(XtrainKeras, YtrainKeras, batch_size=batch_size, nb_epoch=nb_epoch,show_accuracy=True, validation_data=(XtrainKeras, YtrainKeras))
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    #f.write()
    f3=open('AutoEncoderOnPca.txt','wb')
    f3.write(json.dumps(hist.history))
    f3.close()
    Fname=prefix+'2 FineTuning_model='+'.json'
    weightName=prefix+'Fine Tunes Weights_autoencoder_i='+str(i)+'.h5'
    json_string = model.to_json() 
    open(Fname, 'w').write(json_string) 
    model.save_weights(weightName,overwrite=True)


    

if __name__=='__main__':
    train()
