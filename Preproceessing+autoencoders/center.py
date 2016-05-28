import numpy as np
import os
import cv2
import pylab as pl
from os.path import expanduser

home = expanduser("~")
print 'home is ',home
prefix=home+'/gabor/'
src=prefix+'rotate_all_pattern/'
#pl.figure()
i=1
mask1=np.zeros((96,96),np.uint8)
mask2=np.zeros((96,96),np.uint8)
mask1[30:60]=1
#mask2[30:51]=1
'''mask1[39:42,26:70]=0
mask1[57:60,26:70]=0
mask2[30:33,26:70]=0
mask2[48:51,26:70]=0'''

for subdir, dirs, files in os.walk(src):
    print 'subdie is ',subdir
    for file1 in files:
        print 'file number is ',i
        #for i in range(0,len(filters)):
        img=cv2.imread(src+file1,0)
        #img.astype(np.float32)
        fimg = img - img.mean()
        path=prefix+'CenteredWhaleData/'
        if not os.path.exists(path):
            os.makedirs(path)
        imgP=path + file1
        #fimg.astype(np.uint8)
        image_m1= cv2.bitwise_and(fimg,fimg,mask=mask1)
        image_m2= cv2.bitwise_and(fimg,fimg,mask=mask2)
        sum1=image_m1.sum()
        #res=image_m1
        sum2=image_m2.sum()
        #if sum1>sum2:
            #res=image_m1[39:60]
        #else:
            #res=image_m2[30:51]
        res=image_m1[30:60]   
        #pl.subplot(2,2,i)
        #pl.imshow(fimg,cmap='gray')
        #pl.title(file1)
        #i=i+1
        #pl.subplot(2,2,i)
        #pl.imshow(res,cmap='gray')
        #pl.title('masked_'+file1)
        i=i+1
        path2=prefix+'Masked Files/'
        imgF2=path2+file1
        path3=prefix+'Masked Files flipped/'+file1
        imgFlipped=cv2.flip(res,1)
        cv2.imwrite(imgP,fimg)
        cv2.imwrite(imgF2,res)     
        cv2.imwrite(path3,imgFlipped) 
        
#pl.show()
