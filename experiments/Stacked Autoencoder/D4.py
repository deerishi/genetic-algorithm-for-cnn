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




from os.path import expanduser


home = expanduser("~")

prefix=home+'/model Files From AutoEncoder Exp/'

batch_size = 69
nb_classes = 447
nb_epoch = 30





def train():

    model=Sequential()
    
    X_train=np.load(home+'/gabor/numpyFiles/Training Set.npy')
    X_test=np.load(home+'/gabor/numpyFiles/TestSet.npy')
    Y_train=np.load(home+'/gabor/numpyFiles/Training Labels.npy')
    Y_test=np.load(home+'/gabor/numpyFiles/TestSet Labels.npy')
    
    #X_test = X_test.reshape(-1, 1, 30, 96)
    Y_test = np_utils.to_categorical(Y_test, 447)
    
    
    #X_train = X_train.reshape(-1, 1, 30, 96)
    Y_train = np_utils.to_categorical(Y_train, 447)
    
    print("X_test.shape == {};".format(X_test.shape))
    print("Y_test.shape == {};".format(Y_test.shape))
    print("X_test.shape == {};".format(X_train.shape))
    print("Y_test.shape == {};".format(Y_train.shape))
    
    nb_hidden_layers = [len(X_train[0]),700, 500, 300]
    
    XtrainKeras=X_train
    print 'shape of XTrain Keras is ',XtrainKeras.shape
    YtrainKeras=np_utils.to_categorical(Y_train, nb_classes)
    op1=RMSprop(lr=0.01,rho=0.5,epsilon=1e-8)

    X_train_tmp=XtrainKeras
    trained_encoders=[]


    #XtrainKeras=XwhiteTrain.reshape(-1,1,len(XwhiteTrain),len(XwhiteTrain[0]))
    #YtrainKeras=np_utils.to_categorical(Y_train, nb_classes)


    #XtestKeras=X_test.reshape(-1,1,imageWidth,imageHeight)
    #YtestKeras=np_utils.to_categorical(Y_test, nb_classes)
    #X_train_tmp=XtrainKeras

    for n_in, n_out in zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]):
        print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
        # Create AE and training
        ae = Sequential()
        encoder = containers.Sequential([Dense( n_out,input_dim=n_in, activation='sigmoid')])
        decoder = containers.Sequential([Dense( n_in,input_dim=n_out, activation='sigmoid')])
        ae.add(AutoEncoder(encoder=encoder, decoder=decoder,output_reconstruction=False))
        ae.compile(loss='mean_squared_error', optimizer=op1)
        hist=ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=nb_epoch)
        print(hist.history)
        Fname=prefix+'autoencoder_n_in='+str(n_in)+'_n_out= '+str(n_out)+'.json'
        weightName=prefix+'Weights_autoencoder_n_in='+str(n_in)+'_n_out= '+str(n_out)+'.h5'
        json_string = model.to_json() 
        open(Fname, 'w').write(json_string) 
        model.save_weights(weightName,overwrite=True)
        # Store trainined weight
        trained_encoders.append(ae.layers[0].encoder)
        # Update training data
        X_train_tmp = ae.predict(X_train_tmp)
        
        
    #ae1=Sequential()
    #encoder1=containers.Sequential([Dense(len(XwhiteTrain[0])-200,len(XwhiteTrain[0]),activation='sigmoid')])    

    Y_test=np_utils.to_categorical(Y_test, nb_classes)
    #X_test=X_test.reshape(-1,len(X_test[0]))
    print 'shape of X_test  is ',X_test.shape
    print('Fine-tuning')
    sgd=SGD(lr=0.01, momentum=0.5, decay=0., nesterov=False)

    i=1
    model = Sequential()
    for encoder in trained_encoders:
        model.add(encoder)
    model.add(Dense( nb_classes,input_dim=nb_hidden_layers[-1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    hist=model.fit(XtrainKeras, YtrainKeras, batch_size=batch_size, nb_epoch=nb_epoch,show_accuracy=True, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    Fname=prefix+'2 FineTuning_model='+'.json'
    weightName=prefix+'Fine Tunes Weights_autoencoder_i='+str(i)+'.h5'
    json_string = model.to_json() 
    open(Fname, 'w').write(json_string) 
    model.save_weights(weightName,overwrite=True)


    

if __name__=='__main__':
    train()
