from sklearn import svm
import numpy as np
from sklearn.externals import joblib

from sklearn import linear_model

import classifier.msg as msg

SINGLE = 0
MULTIPLE = 1


def ini():
    '''Initialisierung

    Args:


    Returns:

    '''
    global clf
    #clf = svm.SVC(kernel='linear', C = 1.0) # SVM
    clf = linear_model.LogisticRegression() #LR



def train(features, labels):
    '''Klassifizierer trainieren

    Args:
        features (list): Features
        labels (list): Labels

    Returns:

    '''
    global clf
    msg.timemsg("train_shape: {}".format(features.shape))
    clf.fit(features, labels)



def predict(X, mode):
    '''Prediction

    Args:


    Returns:
        prediction (list)

    '''
    global clf
    if (mode == MULTIPLE): # Mehrere Features
        #msg.timemsg("predict_shape: {}".format(X.shape))
        return clf.predict(X)

    if (mode == SINGLE):
        return np.squeeze(np.array(clf.predict(X.reshape(1,-1)))) # Eine Dimension zuviel, diese entfernen

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
