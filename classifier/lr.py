#!/usr/bin/env python

from sklearn import svm
import numpy as np
from sklearn.externals import joblib

from sklearn import linear_model

import classifier.msg as msg
import os

SINGLE = 0
MULTIPLE = 1


def ini(path=None):
    '''initialization

    Args:


    Returns:

    '''
    global clf
    clf = linear_model.LogisticRegression(class_weight='balanced') #LR
    if path is not None:
        if os.path.exists(path):
            clf = joblib.load(path)
            msg.timemsg('Loaded classifier from: {}'.format(path))
        else:
            msg.timemsg('Path to classifier does not exist: {}'.format(path))
    # SVM 
    #clf = svm.SVC(kernel='linear', C = 1.0) # SVM



def train(features, labels, path='clf.pkl'):
    '''train classifier

    Args:
        features (list): Features
        labels (list): Labels

    Returns:

    '''
    global clf
    msg.timemsg("train_shape: {}".format(features.shape))
    msg.timemsg('Start training')
    clf.fit(features, labels)
    msg.timemsg('Finished training')
    try:
        joblib.dump(clf, path)
        msg.timemsg('Dumped classifier')
    except:
        msg.timemsg('Failed to dump classifier!')



def predict(X, mode):
    '''Prediction

    Args:


    Returns:
        prediction (list)

    '''
    global clf
    if (mode == MULTIPLE): # many features
        #msg.timemsg("predict_shape: {}".format(X.shape))
        return clf.predict(X)

    if (mode == SINGLE):
        return np.squeeze(np.array(clf.predict(X.reshape(1,-1)))) # convert to array, one dimension too much, remove it

def save (folder):
    '''Save

    Args:
        folder


    Returns:
        .pkl

    '''
    global clf
    joblib.dump(clf, folder)

def load (folder):
    '''Save

    Args:
        folder


    Returns:
        .pkl

    '''
    global clf
    clf = joblib.load(folder)
