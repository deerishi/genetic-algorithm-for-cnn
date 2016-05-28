#!/usr/bin/env python

import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
 
# /home/drishi/Kaggle Dataset/imgs_head/
img = cv2.imread('/home/drishi/Kaggle Dataset/imgs_head/w_31.jpg')
Z = img.reshape((-1,3))
print 'Z is ',Z
print 'z is ', Z.shape
 
# convert to np.float32
Z = np.float32(Z)
 
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
#K = int(sys.argv[1])
K=4
ret,label,center = cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
print 'label size is ',center.shape
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
 
cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
