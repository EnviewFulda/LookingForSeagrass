#!/usr/bin/env python

import numpy as np

def ini():
    '''initialization

    Args:


    Returns:

    '''
    pass


def accuracy(Yte_predict, Yte):
    '''verification of the correct label with the predicted one

    Args:
        Yte_predict (list): predicted labels (by computer)
        Yte (list): true labels (by human)


    Returns:
        accuracy (list): congruity
    '''
    return np.mean(Yte_predict == Yte)


def ratio(Yte_predict):
    '''Percentage portion of array

    Args:
        Yte_predict (list): predicted labels (by computer)

    Returns:
        accuracy (list): relative part of the labels "1" in the array
    '''
    return np.mean(Yte_predict == 1)
