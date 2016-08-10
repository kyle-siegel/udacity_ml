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

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
sort_keys='../tools/python2_lesson14_keys.pkl'
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30,
                                                                             random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(labels_test, pred)
prec = precision_score(labels_test, pred)
rec = recall_score(labels_test, pred)

print "Accuracy:", acc
print "Predicted POI's:", sum(pred)
print "Size of test set:", len(pred)
print "True positives:", sum([pred[i]==labels_test[i] for i in range(0, len(pred)) if pred[i]])
print "Precision:", prec
print "Recall:", rec



