#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# # reducing size of training dataset
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

# # Support Vector Machine, Linear Kernel
# t0 = time()
# clf = SVC(kernel='linear')
# clf.fit(features_train, labels_train)
# print "training time:", round(time()-t0, 3), "s"

# # Support Vector Machine, rbc Kernel (optimizer)
# for c in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
#     t0 = time()
#     clf = SVC(kernel='rbf', C=c)
#     clf.fit(features_train, labels_train)
#     print "training time:", round(time()-t0, 3), "s"

#     pred = clf.predict(features_test)
#     print "training + predicting time:", round(time()-t0, 3), "s"

#     print accuracy_score(pred, labels_test)

# Support Vector Machine, rbc Kernel (optimized)
t0 = time()
clf = SVC(kernel='rbf', C=10000.0)
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

pred = clf.predict(features_test)
print "training + predicting time:", round(time()-t0, 3), "s"
print accuracy_score(pred, labels_test)

# Predicing for specific email
print 'Element 10, 26, 50: ;', clf.predict([features_test[10], features_test[26], features_test[50]])

# Count predicted 1 (Chris) in test dataset
print 'The count of value 1 (Chris) in test dataset: ', list(pred).count(1)

#########################################################


