#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)

### your code goes here 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import collections


features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                            labels,
                                                                            test_size=0.3,
                                                                            random_state=42)


deciTree = DecisionTreeClassifier()
deciTree.fit(features_train, labels_train)

print('Accuracy score: {}'.format(deciTree.score(features_test, labels_test)))

pred = deciTree.predict(features_test)

print('Prediction value counts for test dataset: {}'.format(collections.Counter(pred)))
print('Total predictions for test dataset: {}'.format(len(pred)))
print('CrossTab/confusion matrix for test dataset:\n{}'.format(confusion_matrix(labels_test, pred)))
print('Precision score for test dataset: {}'.format(precision_score(labels_test, pred)))
print('Recall_score for test dataset: {}'.format(recall_score(labels_test, pred)))


