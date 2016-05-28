import csv
from shutil import copyfile
import cPickle
import numpy as np
from os.path import expanduser
import re
import cv2
from sklearn.utils import shuffle

home = expanduser("~")



def saveFile(dataset,name):
    print 'shape of '+name + ' is ',dataset.shape
    path2=home+'/gabor/numpyFiles/'+name
    np.save(path2,dataset)


ftest2=open(home+'/sample_submission.csv')
testReader2=csv.reader(ftest2)
xTest=[]
#yTest=[]
i=1
for row in testReader2:
    if i==1:
        i=i+1
        continue;
    else:
        if row[0]=='7489':
            continue
        testMap[row[0]]=int(row[2])
        path=home+'/gabor/Masked Files/'+row[0]
        img=cv2.imread(path,0)
        img=img.ravel()
        img=img.astype(np.float32)
        img=img/255; #to normalize the image
        img=img.tolist()
        xTest.append(img)
        #yTest.append(int(row[2]))
        i=i+1
        print 'Testing i is ',i


xTest=np.asarray(xTest)
#yTest=np.asarray(yTest)
#yTest=yTest.T
saveFile(xTest,"Unlabeled Test Set")
#saveFile(yTest,"TestSet Labels")

print 'the shape  of XTrain  ',xTest.shape

