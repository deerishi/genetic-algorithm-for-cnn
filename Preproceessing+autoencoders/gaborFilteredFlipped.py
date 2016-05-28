#!/usr/bin/env python
 
import numpy as np
import os
import cv2
import pylab as pl
import cPickle
from os.path import expanduser

home = expanduser("~")
print 'home is ',home


prefix=home+'/gabor/Masked Files 2'

filters = []
ksize = 9
lamdas=[9]
thetas=[0,5*np.pi /12 , np.pi/2,7*np.pi/12]
for theta in thetas:
    for lamda in lamdas: 
        kern = cv2.getGaborKernel((ksize, ksize), 2.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
        
#img=cv2.imread(root+'raw/'+'file1')
#fimg = cv2.filter2D(img, cv2.CV_8UC3, filters[i])
f=np.asarray(filters)
pl.figure;
for k,im in enumerate(f[:12,:]):
    pl.subplot(4,3,k+1)
    pl.imshow(im.reshape(ksize,ksize), cmap='gray' )
    #pl.xlim(10,20)
    #pl.ylim(10,20)
 
pl.show()


root='/home/drishi/gabor/'
j=1
for subdir, dirs, files in os.walk(root+'Masked Files flipped'):
    print 'subdie is ',subdir
    for file1 in files:
        print 'file name is ',j
        j=j+1
        for i in range(0,len(filters)):
            img=cv2.imread(root+'Masked Files flipped/'+file1,0)
            #img.astype(np.float32)
            fimg = cv2.filter2D(img, cv2.CV_8UC3, filters[i])
            path=root+'filteredFlipped/filter '+str(i)+'/'
            if not os.path.exists(path):
                os.makedirs(path)
            imgP=path + file1
            #fimg.astype(np.uint8)
            cv2.imwrite(imgP,fimg)
            #pl.figure;
            
            #pl.subplot(4,3,i+1)
            #pl.imshow(fimg, cmap='gray' )
            #pl.title(file1)
            #pl.xlim(10,20)
            #pl.ylim(10,20)
         
        #pl.show()     
