import Modules
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

pathname = 'E:/00. University/0. TA/PR/assignments/Assignment_1/data/'

[image, array] = Modules.read_image(pathname, 'HSI_crop.tif', normalize=False)
plt.figure(), plt.imshow(array[:, :, 90].T, cmap='gray'), plt.show()


[roi, labels] = Modules.read_roi(pathname, 'mixed_ROI.tif', separate=True, percent=0.2)
plt.figure(), plt.imshow(roi.T, cmap='jet'), plt.show()


plt.figure()
[Train, labels] = Modules.read_roi(pathname, 'PySeparateTrain.tif')
plt.subplot(211), plt.imshow(Train.T, cmap='jet')
[Test, labels] = Modules.read_roi(pathname, 'PySeparateTest.tif')
plt.subplot(212), plt.imshow(Test.T, cmap='jet')
plt.show()

# appropriate array for machine learning
data = np.reshape(array, (array.shape[0]*array.shape[1], array.shape[2]), order='F')
train = np.reshape(Train, (Train.shape[0]*Train.shape[1], 1), order='F')
test = np.reshape(Test, (Test.shape[0]*Test.shape[1], 1), order='F')

# select N random features from all the features (bands, here 144)
N = 80
selectedBands = np.random.permutation(np.arange(array.shape[2]))[:N]
print(selectedBands)
selectedArray = data[:, selectedBands]

# calculate mean/covariance/inverse/det for each class in each feature
Means = np.zeros((len(labels), N))
Covariance = np.zeros((N, N, len(labels)))
InvCov = np.zeros_like(Covariance)
Det = np.zeros((len(labels), 1))
SlogDet = np.zeros((len(labels), 1))

for l in labels:
    Means[l-1, :] = np.mean(selectedArray[np.where(train == l), :], axis=1)[0, :]
    Covariance[:, :, l-1] = np.cov(selectedArray[np.where(train == l), :][0], rowvar=False)
    InvCov[:, :, l-1] = np.linalg.pinv(Covariance[:, :, l-1])
    # Det[l-1] = np.linalg.det(Covariance[:, :, l-1])
    SlogDet[l-1] = np.linalg.slogdet(Covariance[:, :, l-1])[1]

# calculate discriminant function
# 2443 * 150 = 366450
G_x = np.zeros((selectedArray.shape[0], len(labels)))
t0 = time.time()
for i in range(1, 151):
    for c in labels:
        G_x[i*2443-2443:i*2443, c-1] = -0.5*np.diag(np.matmul(np.matmul(np.subtract(selectedArray[i*2443-2443:i*2443, :]
                                                                                    , Means[c - 1, :]),
                                                        np.atleast_2d(InvCov[:, :, c - 1])),
                                                        np.subtract(selectedArray[i*2443-2443:i*2443, :],
                                                                    Means[c - 1, :]).T)) - 0.5 * (SlogDet[l-1])

# np.log(Det[c-1]
print(time.time() - t0)

# t0 = time.time()
#
# for i in range(G_x.shape[0]):
#     for c in labels:
#         G_x[i, c-1] = -0.5 * np.matmul(np.matmul(np.atleast_2d(selectedArray[i, :] - Means[c-1, :]),
#                                                  np.atleast_2d(InvCov[:, :, c-1])),
#                                        np.atleast_2d(selectedArray[i, :] - Means[c-1, :]).T) - 0.5 * np.log(Det[c-1])
#
# print(time.time() - t0)
# plt.figure(), plt.imshow(np.reshape(np.argmax(G_x, axis=1)+1, (349, 1050)), cmap='jet'), plt.show()


t0 = time.time()
GNB = GaussianNB()
GNB.fit(selectedArray[np.where(train != 0)[0]], train[train != 0])
G_x_skl1 = GNB.predict(selectedArray)
G_x_skl2 = GNB.predict_log_proba(selectedArray)
print(time.time() - t0)

plt.figure()
plt.subplot(211), plt.title('My result')
plt.imshow(np.reshape(np.argmax(G_x, axis=1)+1, (349, 1050)), cmap='jet')
plt.subplot(212), plt.title('SKLearn result')
plt.imshow(np.reshape(G_x_skl1, (349, 1050)), cmap='jet')
plt.show()

plt.figure()
for l in labels:
    plt.subplot(3, 3, l)
    plt.imshow(np.reshape(G_x_skl2[:, l-1], (349, 1050))**.55)
plt.show()

print('\t\t\t\t---- Accuracy assessment ----')
print(classification_report(test[test != 0], G_x_skl1[np.where(test != 0)[0]]))
print('\t\tThe classification accuracy is reported above for sklearn classifier ^\n')
print(accuracy_score(test[test != 0], (np.argmax(G_x, axis=1)+1)[np.where(test != 0)[0]]))
print('\t\tThy classification accuracy is reported above ^\n')
print('\t\t\t\t---- Finish ----')
