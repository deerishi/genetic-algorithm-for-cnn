#!/usr/bin/env python

import numpy as np
import cv2
import sys
import os
from matplotlib import pyplot as plt
from os.path import expanduser

home=expanduser("~")
prefix=home+'/gabor/'

# /home/drishi/Kaggle Dataset/imgs_head/

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
K = int(sys.argv[1])

root=home+'/gabor/'
j=1
for subdir, dirs, files in os.walk(root+'Masked Files'):
    #print 'subdie is ',subdir
    for file1 in files:
        #print 'file name is ',j
        j=j+1
        img=cv2.imread(root+'Masked Files/'+file1,0)
        Z = img.reshape((-1,1))
        Z = np.float32(Z)
        #K=4
        ret,label,center = cv2.kmeans(Z,K,criteria,25,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        des=root+'KmeansOnMasked/'+file1
        cv2.imwrite(des,res2)
