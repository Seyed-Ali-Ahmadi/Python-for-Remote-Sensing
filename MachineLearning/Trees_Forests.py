import Modules
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz
import warnings

warnings.filterwarnings('ignore')

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
selectedBands = [129, 52, 48, 73, 128, 92, 71, 23, 87, 111, 142, 141, 27,
                 58, 115, 99, 84, 132, 13, 35, 77, 89, 113, 102, 36, 38,
                 131, 39, 94, 5, 66, 2, 134, 51, 96, 24, 114, 121, 120, 46]
print(selectedBands)
selectedArray = data[:, selectedBands]

class_names = ['healthy veg', 'stressed veg', 'synthetic grass', 'soil', 'buildings',
               'roads', 'parking lot', 'tennis court', 'running track']
#
# # Decision Tree
# tree = DecisionTreeClassifier(criterion='gini',     # criterion='entropy
#                               splitter='best',      # splitter='random'
#                               max_depth=5,          # max_depth=None
#                               max_features='auto',  # max_features='sqrt', 'log2', None
#                               random_state=0)
# t0 = time.time()
# tree.fit(selectedArray[np.where(train != 0)[0]], train[train != 0])
# print(time.time() - t0)
# classified = tree.predict(selectedArray[np.where(test != 0)[0]])
# plt.figure(), plt.plot(tree.feature_importances_), plt.show()
# print(accuracy_score(classified, test[test != 0]))
# # How to export
# data = export_graphviz(tree, out_file=None, class_names=class_names)
# graph = graphviz.Source(data)
# graph.render('tree')
#
# t0 = time.time()
# classified_image1 = tree.predict(selectedArray)
# print(time.time() - t0)
# plt.figure()
# plt.imshow(np.reshape(classified_image1, (1050, 349), order='F').T, cmap='jet'), plt.show()
#
# #########################
# tree = DecisionTreeClassifier(criterion='gini',     # criterion='entropy
#                               splitter='best',      # splitter='random'
#                               max_features='auto',  # max_features='sqrt', 'log2', None
#                               random_state=0)
# tree.fit(selectedArray[np.where(train != 0)[0]], train[train != 0])
# print(accuracy_score(classified, test[test != 0]))
# classified_image = tree.predict(selectedArray)
# plt.figure()
# plt.imshow((np.reshape(classified_image, (1050, 349), order='F').T -
#            np.reshape(classified_image1, (1050, 349), order='F').T) != 0, cmap='gray')
# plt.show()
#
#
#
#
# Random Forests
n_estimators = 100
t_start = time.time()
forest = RandomForestClassifier(n_estimators=n_estimators,
                                criterion='gini',
                                max_depth=5,
                                max_features='sqrt',    # 'log2', None, 'sqrt'
                                n_jobs=-1,
                                warm_start=False,
                                random_state=0)
forest.fit(selectedArray[np.where(train != 0)[0]], train[train != 0])
t_fit = time.time() - t_start
print('fitting time = {}'.format(t_fit))
t_start = time.time()
classified_rf = forest.predict(selectedArray)
t_pred = time.time() - t_start
print('prediction time = {}'.format(t_pred))
plt.figure()
plt.title('Random Forest with {0} estimators'.format(n_estimators))
plt.imshow(np.reshape(classified_rf, (1050, 349), order='F').T, cmap='jet')
plt.xlabel('accuracy is {}'.format(accuracy_score(forest.predict(selectedArray[np.where(test != 0)[0]]),
                                                  test[test != 0])))
plt.tick_params(
    axis='x',           # changes apply to the x-axis
    which='both',       # both major and minor ticks are affected
    bottom=False,       # ticks along the bottom edge are off
    top=False,          # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off
plt.show()

# study errors
m = 0
oob_error = np.zeros(160)
t_start = time.time()
for i in range(15, 175):
    forest = RandomForestClassifier(n_estimators=i,
                                    criterion='gini',
                                    max_depth=5,
                                    max_features='sqrt',    # 'log2', None
                                    oob_score=True,
                                    n_jobs=-1,
                                    warm_start=False,
                                    random_state=0)
    forest.fit(selectedArray[np.where(train != 0)[0]], train[train != 0])
    oob_error[m] = 1 - forest.oob_score_
    m += 1

t_fit = time.time() - t_start
print('fitting time = {}'.format(t_fit))
plt.figure(), plt.plot(range(15, 175), oob_error)
plt.title('Out-Of-Bag Errors OOB')
plt.show()

std = np.std([T.feature_importances_ for T in forest.estimators_], axis=0)
indices = np.argsort(forest.feature_importances_)[::-1]
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(selectedArray.shape[1]), forest.feature_importances_[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(selectedArray.shape[1]), indices)
plt.xlim([-1, selectedArray.shape[1]])
plt.show()

classified = forest.predict(selectedArray[np.where(test != 0)[0]])
print(accuracy_score(classified, test[test != 0]))


