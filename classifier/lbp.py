#!/usr/bin/env python

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq


SINGLE = 0
MULTIPLE = 1


def ini():
    '''initialization

    Args:


    Returns:

    '''


def features(X, mode):
    '''extract features

    Args:
        X (list):  Patch or Patches
        mode: SINGLE, MULTIPLE

    Returns:
        features (list)
    '''

    radius = 3
    n_points = 8 * radius
    METHOD = 'uniform'

    if (mode == MULTIPLE): # may features
        listFeatures = None
        for Patch in X:
            gray_image = cv2.cvtColor(Patch, cv2.COLOR_BGR2GRAY)

            feat = local_binary_pattern(gray_image, n_points, radius, METHOD)
            tmp = feat.ravel()
            tmp = tmp / np.max(tmp)
            hist = np.histogram(tmp, bins=100)[0]

            if listFeatures is None:
                listFeatures = hist
            else:
                listFeatures = np.vstack((listFeatures, hist))
        return listFeatures

    if (mode == SINGLE): # single feature
        gray_image = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
        feat = local_binary_pattern(gray_image, n_points, radius, METHOD)

        x = itemfreq(feat.ravel())
        hist = x[:, 1]/sum(x[:, 1])

        return hist



def featurelist(path_train_set):
    '''load training patches and extract features

    Args:
        path_train_set (list): paths to training patches


    Returns:
        features (list)

    '''
    listFeatures = []

    for index, Path in enumerate(path_train_set):
        loaded_patch = cv2.imread(Path)
        listFeatures.append(features(loaded_patch, SINGLE))

    return listFeatures
