#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
import time

sys.path.append("../tools/")
from sklearn.metrics import accuracy_score
from email_preprocess import preprocess
from sklearn.svm import SVC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
### your code goes here ###
clf_rbf = SVC(kernel='rbf', C=10000)
t0 = time.time()
clf_rbf.fit(features_train, labels_train)
print("training time:", round(time.time() - t0, 3), "s")

t0 = time.time()
pred = clf_rbf.predict(features_test)
print("testing time:", round(time.time() - t0, 3), "s")

#########################################################
accuracy = accuracy_score(pred, labels_test)
print('Accuracy with rbf and C=10000: ', accuracy)

chris = 0
sara = 0
for i in pred:
    if i == 1:
        chris += 1
    elif i == 0:
        sara += 1
print(chris)
print(sara)
#########################################################
