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

WHALE_TRAIN_DATA = "/home/drishi/train_data_128.csv"
WHALE_TEST_DATA = "/home/drishi/test_data_128.csv"

Xtrain_file='/home/drishi/trainingData128_file'
Ytrain_file='/home/drishi/trainingLabel128_file'
Xtest_file='/home/drishi/testData128_file'
Ytest_file='/home/drishi/testLabel128_file'

covxFile='/home/drishi/projectFiles/CocX128.npy'
UFromSVD_file='/home/drishi/projectFiles/UfromSVD128_file.npy'
VFromSVD_file='/home/drishi/projectFiles/VfromSVD128_file.npy'
SFromSVD_file='/home/drishi/projectFiles/SfromSVD128_file.npy'
WhitenedData128='/home/drishi/projectFiles/WhitenedData128'
WhitenedTestData128='/home/drishi/projectFiles/WhitenedTestData128'

nb_hidden_layers = [192*192, 100*50, 1000, 192]
batch_size = 128
nb_classes = 447
nb_epoch = 12
imageWidth=128
imageHeight=128

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

def whiten():
    X_train=np.load(Xtrain_file+'.npy')
    Y_train=np.load(Ytrain_file+'.npy')
    X_test=np.load(Xtest_file+'.npy')
    Y_test=np.load(Ytest_file+'.npy')
    print 'shape os X_train is ',X_train.shape
    print 'shape os y_train is ',Y_train.shape
    print 'shape os X_test  is ',X_test.shape
    print 'shape os y_test is ',Y_test.shape

    #First Step of Whitening is Centering the datasets
    Xtrain_total=np.concatenate((X_train,X_test),axis=0)
    Ytrain_total=np.concatenate((Y_train,Y_test),axis=0)

    print 'shape os Xtrain_total is ',Xtrain_total.shape
    print 'shape os Ytrain_total is ',Ytrain_total.shape

    XCentered_Train=np.zeros(X_train.shape,dtype=np.float32)
    XCentered_Test=np.zeros(X_test.shape,dtype=np.float32)
    mean_matrix=X_train.mean(0)
    print '-->Centering Now'
    for i in range(0,len(X_train[0])):
        XCentered_Train[:,i]=X_train[:,i]-mean_matrix[i]
        XCentered_Test[:,i]=X_test[:,i]-mean_matrix[i]
    #now centering the test set
    print 'centered ,now finding covariance'
    #NOw find the covariance Matrix

    covX=np.load(covxFile)
    #covX=np.cov(XCentered_Train.T) #In numpy each row is taken as a variable
    print ' covariance found with size ',covX.shape, 'now svd'
    #Compute the SIngular Value decomposition of the covariance matrix
    #np.save(covxFile,covX)

    #[u,s,v]=np.linalg.svd(covX)
    #np.save(UFromSVD_file,u)
    #np.save(VFromSVD_file,v)
    #np.save(SFromSVD_file,s)
    u=np.load(UFromSVD_file)
    v=np.load(VFromSVD_file)
    s=np.load(SFromSVD_file)
    print ' u found with size ',u.shape
    print ' v found with size ',v.shape
    print ' s found with size ',s.shape
    print 'svd complete now cumsome'
    latentVariance=np.cumsum(s)/sum(s)
    #use binary search to find the expected value of good variance
    #start=len(latentVariance)/2
    start=0
    print 'latentVariance[',start,'] is ',latentVariance[start]
    while(latentVariance[start]<1):
        #print 'latentVariance[',start,'] is ',latentVariance[start]
        #start=(start+len(latentVariance))/2
        start=(start)+1


    #start=len(latentVariance)/4
    print 'final latentVariance[',start-1,'] is ',latentVariance[start-1]
    print 'final latentVariance[',start,'] is ',latentVariance[start]
    #princomps=np.zeros((len(u),start+1))
    princomps=u[:,0:(start)]
    #Transform to Principle component space
    XNewTrain=np.dot(princomps.T,XCentered_Train.T) # This is our reduced dimension but its the transpose.
    XNewTest=np.dot(princomps.T,XCentered_Test.T)

    XwhiteTrain=np.zeros(XNewTrain.shape,dtype=np.float32)
    XwhiteTest=np.zeros(XNewTest.shape,dtype=np.float32)
    #NOw we do whitening
    NewEigen=s[0:(start)]
    epsilon=0.0001
    D=np.diag(1./(np.sqrt((np.diag(NewEigen) + epsilon))))
    print 'size of the diagonal matrix is ',D.shape

    for i in range(0,start):
        XwhiteTrain[i]=XNewTrain[i]*D[i]
        XwhiteTest[i]=XNewTest[i]*D[i]

    print 'the shape of XwhiteTrain is ',XwhiteTrain.shape
    print 'the shape of XwhiteTest is ',XwhiteTest.shape
    XwhiteTrain=XwhiteTrain.T #converting into our row column form
    XwhiteTest=XwhiteTest.T

    np.save(WhitenedData128,XwhiteTrain)
    np.save(WhitenedTestData128,XwhiteTest)
    print 'the shape of XwhiteTrain is ',XwhiteTrain.shape
    print 'the shape of XwhiteTest is ',XwhiteTest.shape
    #now we train and model  the autoencoder
    model=Sequential()

    XtrainKeras=XwhiteTrain.reshape(-1,1,len(XwhiteTrain),len(XwhiteTrain[0]))
    YtrainKeras=np_utils.to_categorical(Y_train, nb_classes)

    ''''
    XtestKeras=X_test.reshape(-1,1,imageWidth,imageHeight)
    YtestKeras=np_utils.to_categorical(Y_test, nb_classes)
    X_train_tmp=XtestKeras
    for n_in, n_out in zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]):
        print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
        # Create AE and training
        ae = Sequential()
        encoder = containers.Sequential([Dense(n_in, n_out, activation='sigmoid')])
        decoder = containers.Sequential([Dense(n_out, n_in, activation='sigmoid')])
        ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
                           output_reconstruction=True, tie_weights=True))
        ae.compile(loss='mean_squared_error', optimizer='rmsprop')
        ae.fit(X_train_tmp, XtestKeras, batch_size=batch_size, nb_epoch=nb_epoch)
        # Store trainined weight
        trained_encoders.append(ae.layers[0].encoder)
        # Update training data
        X_train_tmp = ae.predict(X_train_tmp)'''
if __name__=='__main__':
    whiten()


