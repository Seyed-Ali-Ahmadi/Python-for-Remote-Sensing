import Modules
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.kdtree import KDTree
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


pathname = 'E:/00. University/0. TA/PR/assignments/Assignment_1/data/'

[image, array] = Modules.read_image(pathname, 'HSI_crop.tif', normalize=True)
# [roi, labels] = Modules.read_roi(pathname, 'mixed_ROI.tif', separate=True, percent=0.2)
[Train, labels] = Modules.read_roi(pathname, 'PySeparateTrain.tif')
[Test, labels] = Modules.read_roi(pathname, 'PySeparateTest.tif')

data = np.reshape(array, (array.shape[0]*array.shape[1], array.shape[2]), order='F')
train = np.reshape(Train, (Train.shape[0]*Train.shape[1], 1), order='F')
test = np.reshape(Test, (Test.shape[0]*Test.shape[1], 1), order='F')

k = 5
N = 10
selectedBands = np.random.permutation(np.arange(array.shape[2]))[:N]
print(selectedBands)
selectedArray = data[:, selectedBands]

train_non_0 = train[np.where(train != 0)[0]]
test_non_0 = test[np.where(test != 0)[0]]
selectedTrain = selectedArray[np.where(train != 0)[0], :]
selectedTest = selectedArray[np.where(test != 0)[0], :]

#######################################################################################################################
# Spatial KD-tree method
t0 = time.time()

tree = KDTree(selectedTrain)
print(time.time() - t0)

found = np.zeros((selectedTest.shape[0], k), dtype=np.int)
# Klabels = np.zeros_like(found)
kNN = np.zeros((selectedTest.shape[0], 1), dtype=np.int)

for e in range(selectedTest.shape[0]):
    found[e, :] = tree.query(selectedTest[e, :], k=k)[1]
    # Klabels[e, :] = train_non_0[found[e, :]].ravel()
    kNN[e] = np.argmax(np.bincount(train_non_0[found[e, :]].ravel()))

print('from scratch knn method')
print(time.time() - t0)
print(accuracy_score(test_non_0, kNN.ravel()))
print(classification_report(test_non_0, kNN.ravel()))

#######################################################################################################################
# Scikit-Learn distance method (Actually it's not a method. It is using another method itself.)
t0 = time.time()

eDist = euclidean_distances(selectedTest, selectedTrain)
sorted_eDist = np.argsort(eDist, axis=1)
knn_a = np.reshape(train_non_0[sorted_eDist][:, :5, :], (sorted_eDist.shape[0], k))
kNN = np.zeros((selectedTest.shape[0], 1), dtype=np.int)

for i in range(knn_a.shape[0]):
    kNN[i] = np.argmax(np.bincount(knn_a[i, :]))

print('sklearn distance method')
print(time.time() - t0)
print(accuracy_score(test_non_0, kNN.ravel()))
print(classification_report(test_non_0, kNN.ravel()))

#######################################################################################################################
# Scikit-Learn kNN method
t0 = time.time()

kNN = np.zeros((selectedTest.shape[0], 1), dtype=np.int)
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(selectedTrain, train_non_0.ravel())
kNN = clf.predict(selectedTest)

print('sklearn knn method')
print(time.time() - t0)
print(accuracy_score(test_non_0, kNN.ravel()))
print(classification_report(test_non_0, kNN.ravel()))

#######################################################################################################################



