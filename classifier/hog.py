#!/usr/bin/env python

import cv2
import numpy as np

SINGLE = 0
MULTIPLE = 1


def ini():
    '''initialization

    Args:


    Returns:

    '''
    global hogobj

    winSize = (32,32)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64

    hogobj = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)


def features(X, mode):
    '''Feature exreaction

    Args:
        X (list):  Patch or Patches
        mode: SINGLE, MULTIPLE

    Returns:
        features (list)
    '''
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)

    if (mode == MULTIPLE): # many features
        listFeatures = []


        for Patch in X:
            gray_image = cv2.cvtColor(Patch, cv2.COLOR_BGR2GRAY)
            feat = hogobj.compute(gray_image,winStride,padding,locations)
            feat = np.squeeze(np.array(feat)) # remove one dimension
            listFeatures.append(feat)
        return listFeatures

    if (mode == SINGLE): # one feature
        gray_image = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
        feat = hogobj.compute(gray_image,winStride,padding,locations)
        feat = np.squeeze(np.array(feat)) # remove one dimension
        return feat



def featurelist(path_train_set):
    '''load training patches and extract features

    Args:
        path_train_set (list): path to patches


    Returns:
        features (list)

    '''
    listFeatures = []

    for index, Path in enumerate(path_train_set):
        loaded_patch = cv2.imread(Path)
        listFeatures.append(features(loaded_patch, SINGLE))

    return listFeatures
