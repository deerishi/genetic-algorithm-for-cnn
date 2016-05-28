from sklearn.cluster import DBSCAN
import numpy as np
import cv2
from collections import Counter
from os.path import expanduser

home=expanduser("~")
prefix=home+'/gabor/'
src=prefix+'filtered2/w_0.jpg'

img=cv2.imread(src,0)
Z=img.reshape((-1,1))
db=DBSCAN(eps=0.8,min_samples=15).fit(Z)
labels=db.labels_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
import matplotlib.pyplot as plt
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Black removed and is used for noise instead.
unique_labels = set(labels)
X=Z
colors = plt.cm.Spectral(len(unique_labels))
l2=[]
for i in unique_labels:
    l2.append(i)
map1={}
print 'colors is ',len(colors),' and l2 is ',len(l2)
for i in range(0,len(l2)):
    print 'map is ',i
    map1[l2[i]]=colors[i]
'''colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)'''

for k, col in zip(unique_labels, colors):
    row=int(i/21)
    col=i%96
    plt.plot(row, col, markerfacecolor=map1[labels[i]])    
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
print 'labels are ',(labels)
print 'set of labels are ',Counter(labels)

