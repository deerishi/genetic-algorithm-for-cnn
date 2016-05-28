import csv
from shutil import copyfile
import cPickle
import numpy as np
from os.path import expanduser
import re
import cv2
from sklearn.utils import shuffle

home = expanduser("~")


      
folderList=[home+'/gabor/Masked Files/',home+'/gabor/Masked Files flipped/',home+'/gabor/filtered/filter 0/',home+'/gabor/filtered/filter 1/',home+'/gabor/filtered/filter 2/',home+'/gabor/filtered/filter 3/',home+'/gabor/filteredFlipped/filter 0/',home+'/gabor/filteredFlipped/filter 1/',home+'/gabor/filteredFlipped/filter 2/',home+'/gabor/filteredFlipped/filter 3/']    
def saveFile(dataset,name):
    print 'shape of '+name + ' is ',dataset.shape
    path2=home+'/gabor/numpyFiles/'+name
    np.save(path2,dataset)
    



ftrain=open(home+'/train_lut.csv')
ftest=open(home+'/test_lut.csv')
trainReader=csv.reader(ftrain)
testReader=csv.reader(ftest)
i=1 
trainMap={}
testMap={}
xTrain=[]
yTrain=[]


for row in trainReader:
    if i==1:
        i=i+1
        continue;
    else:
        if row[0]=='7489':
            continue
        trainMap[row[0]]=row[2]
        for j in range(0,len(folderList)):
            path=folderList[j]+'w_'+row[0]+'.jpg'
            img=cv2.imread(path,0)
            img=img.ravel()
            img=img.astype(np.float32)
            img=img/255; #to normalize the image
            img=img.tolist()
            xTrain.append(img)
            yTrain.append(int(row[2]))
        i=i+1
        print 'Training i is ',i
       
xTrain=np.asarray(xTrain)
yTrain=np.asarray(yTrain)
yTrain=yTrain.T
xTrain, yTrain = shuffle(xTrain, yTrain, random_state=42)
saveFile(xTrain,"Training Set")
saveFile(yTrain,"Training Labels")

            
print 'the shape  of XTrain and yTrain ',xTrain.shape,' and ',yTrain.shape
xTestMasked=[]
yTestMasked=[]
i=1
for row in testReader:
    if i==1:
        i=i+1
        continue;
    else:
        if row[0]=='7489':
            continue
        testMap[row[0]]=int(row[2])
        for j in range(0,len(folderList)):
            path=folderList[j]+'w_'+row[0]+'.jpg'
            img=cv2.imread(path,0)
            img=img.ravel()
            img=img.astype(np.float32)
            img=img/255; #to normalize the image
            img=img.tolist()
            xTestMasked.append(img)
            yTestMasked.append(int(row[2]))
        i=i+1
        print 'Testing 1 i is ',i

xTestMasked=np.asarray(xTestMasked)
yTestMasked=np.asarray(yTestMasked)
yTestMasked=yTestMasked.T            
xTrainFull=np.concatenate((xTrain,xTestMasked),0)
yTrainFull=np.concatenate((yTrain,yTestMasked),0)
xTrainFull, yTrainFull = shuffle(xTrainFull, yTrainFull, random_state=42)
saveFile(xTrainFull,"TraingSetFull")
saveFile(yTrainFull,"Training Labels FUll")

print 'the shape  of XTrain and yTrain ',xTrainFull.shape,' and ',yTrainFull.shape

ftest2=open(home+'/test_lut.csv')
testReader2=csv.reader(ftest2)
xTest=[]
yTest=[]
i=1
for row in testReader2:
    if i==1:
        i=i+1
        continue;
    else:
        if row[0]=='7489':
            continue
        testMap[row[0]]=int(row[2])
        path=home+'/gabor/Masked Files/'+'w_'+row[0]+'.jpg'
        img=cv2.imread(path,0)
        img=img.ravel()
        img=img.astype(np.float32)
        img=img/255; #to normalize the image
        img=img.tolist()
        xTest.append(img)
        yTest.append(int(row[2]))
        i=i+1
        print 'Testing i is ',i


xTest=np.asarray(xTest)
yTest=np.asarray(yTest)
yTest=yTest.T
saveFile(xTest,"TestSet")
saveFile(yTest,"TestSet Labels")

print 'the shape  of XTrain and yTrain ',xTest.shape,' and ',yTest.shape 


            
        
