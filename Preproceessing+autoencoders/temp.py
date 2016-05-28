import numpy as np
import os
import cv2
import pylab as pl
from os.path import expanduser

home = expanduser("~")
print 'home is ',home
prefix=home+'/gabor/'
src=prefix+'Masked Files/'
pl.figure()
'''i=cv2.imread(src,0)
i1,i2,i3=cv2.split(i)
pl.figure()

pl.subplot(1,4,1)
pl.imshow(i1,cmap='gray')

pl.subplot(1,4,2)
pl.imshow(i2,cmap='gray')

pl.subplot(1,4,3)
pl.imshow(i3,cmap='gray')

pl.subplot(1,4,1)
pl.imshow(i,cmap='gray')
f=cv2.flip(i,1)
#f[38:57]=1
#res = cv2.bitwise_and(i,i,mask = f)
pl.subplot(1,4,2)
pl.imshow(f,cmap='gray')
'''
i=1
for subdir, dirs, files in os.walk(src):
    print 'subdie is ',subdir
    for file1 in files:
        print 'file number is ',file1
        if i==101:
            pl.show()
            i=1
            pl.figure()
        
        path=src+file1
        img=cv2.imread(path)
        pl.subplot(10,10,i)
        pl.imshow(img,cmap='gray')
        pl.title(file1)
        i=i+1
        
        
        
        
pl.show()
