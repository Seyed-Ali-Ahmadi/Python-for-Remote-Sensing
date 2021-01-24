import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from osgeo import gdal, gdal_array
import scipy.io as sio
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import time

t0 = time.time()
#########################################
# LOAD                                                                                                  <<<<<<<<<<<<<<<<
# f = open('./Accuracies/_16.txt', 'w')
# Class1 = np.load('./results/final/12.npy')
final = sio.loadmat('./results/final/15.mat')
Class1 = final['ALL']

# Class1 = Class1[:, 0:1330]

plt.imshow(Class1, cmap='jet')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.tight_layout()

print(np.unique(Class1))
# plt.show()

Class1 = sp.medfilt(Class1, [3, 3])
plt.figure()
plt.imshow(Class1, cmap='jet')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.tight_layout()

# plt.figure()
# plt.imshow(MED, cmap='jet')
# plt.show()
#########################################
# VALIDATION                                                                                            <<<<<<<<<<<<<<<<
gdal.UseExceptions()
gdal.AllRegister()

roi_ds = gdal.Open('./TestROI.tif', gdal.GA_ReadOnly)
roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

# roi = roi[:, 0:1330]

labels = np.unique(roi[roi > 0])
CA = np.zeros_like(labels, np.float32)

for i in range(len(labels)):
    A = Class1[roi == labels[i]]
    S = np.sum(A != i)
    CA[i] = 100*(len(A) - S)/len(A)
    plt.figure()
    A = Class1 == i
    plt.imshow(A)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.tight_layout()
    # writing_accuracy = str(CA[i]) + '\n'
    # f.write(writing_accuracy)

print(CA.T)
OA = np.sum(CA)/len(labels)
# writing_accuracy = '\n' + str(OA) + '\n'
# f.write(writing_accuracy)
print(OA)
# f.close()
plt.show()
t1 = time.time()
print('execution time has become: ' + str(t1-t0) + '  seconds')
####################################################
# Confusion Matrix sklearn
Test = np.zeros([0, 1], np.int8)
LA = np.zeros([0, 1], np.float32)

for c in range(len(labels)):
    CLS = Class1[roi == labels[c]]
    CLS = np.atleast_2d(CLS).T
    print(CLS.shape)
    L = roi[roi == labels[c]]
    L = np.atleast_2d(L).T
    LA = np.vstack((LA, L))
    Test = np.vstack((Test, CLS))

X = Test
Y = LA
# plt.figure()
# plt.plot(Test)
# plt.plot(LA, color='r')
# plt.show()

class_names = ['healthy grass', 'stressed grass', 'synthetic grass',
               'tree', 'soil', 'water', 'residential', 'commercial',
               'road', 'highway', 'railway', 'parking lot 1',
               'parking lot 2', 'tennis court', 'running track']

ConfMat = confusion_matrix(Test, LA)
ConfMat = ConfMat[0:15, 15:30]
fig = plt.imshow(ConfMat, cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = range(len(class_names))
plt.xticks(tick_marks, class_names, rotation=90)
plt.yticks(tick_marks, class_names)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(i, j, str(int(ConfMat[i, j])), horizontalalignment='center', fontsize=7)
plt.show()

#####################################################
# Classification Report
print(Test)
for i in range(len(LA)):
    for j in range(len(labels)):
        if LA[i] == labels[j]:
            LA[i] = j
print(LA)
print(classification_report(LA, Test))
overall_acc = accuracy_score(LA, Test)
print('Overall Accuracy (recall) is:   ', str(overall_acc))

# overall_acc = np.sum(np.diag(ConfMat)) / np.sum(ConfMat)
# print(np.diag(ConfMat))
# print(np.sum(ConfMat, axis=1))
class_acc = np.atleast_2d(np.divide(np.diag(ConfMat), np.sum(ConfMat, axis=1))).T
# print(class_acc)
print('Average Accuracy (precision) is:   ', np.mean(class_acc))
print('Cohen''s kappa (kappa coefficient) is:   ', cohen_kappa_score(Test, LA))

