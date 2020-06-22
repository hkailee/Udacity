#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
# ### points mixed together--separate them so we can give them different colors
# ### in the scatterplot and identify them visually
# grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
# bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
# grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
# bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


# #### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
### your code goes here ###
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np


for i in range(1,5):
    t0 = time()
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=i), learning_rate=1.0, n_estimators=200, random_state=0)
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"

    pred = clf.predict(features_test)
    print "training + predicting time:", round(time()-t0, 3), "s"

    print accuracy_score(pred, labels_test)

# KNN
t0 = time()
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
print "training + predicting time:", round(time()-t0, 3), "s"

# print the accuracy and display the decision boundary
print 'Accuracy = {0}'.format(clf.score(features_test, labels_test))
prettyPicture(clf, features_test, labels_test)

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
