import Modules
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

pathname = 'E:/00. University/0. TA/PR/assignments/Assignment_1/data/'
[image, array] = Modules.read_image(pathname, 'HSI_crop.tif', normalize=False)
plt.figure(), plt.imshow(array[:, :, 90].T, cmap='gray'), plt.show()

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
N = 40
# selectedBands = np.random.permutation(np.arange(array.shape[2]))[:N]
selectedBands = [129,52,48,73,128,92,71,23,87,111,142,141,27,
                 58,115,99,84,132,13,35,77,89,113,102,36,38,
                 131,39,94,5,66,2,134,51,96,24,114,121,120,46]
print(selectedBands)
selectedArray = data[:, selectedBands]

# Making AdaBoost
bdt_real = AdaBoostClassifier(
                                DecisionTreeClassifier(max_depth=3),
                                n_estimators=500,
                                learning_rate=1)

bdt_real.fit(selectedArray[np.where(train != 0)[0]], train[train != 0])
real_test_errors = []

for real_test_predict in \
        bdt_real.staged_predict(selectedArray[np.where(test != 0)[0]]):
    real_test_errors.append(1. - accuracy_score(real_test_predict, test[test != 0]))

n_trees_real = len(bdt_real)
real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]

classified = bdt_real.predict(selectedArray[np.where(test != 0)[0]])
score = accuracy_score(classified, test[test != 0])
print(score)

plt.figure(figsize=(15, 5))
plt.plot(range(1, n_trees_real + 1),
         real_test_errors, c='black',
         linestyle='dashed', label='SAMME.R')
plt.show()


