#!/usr/bin/env python

import numpy as np

def ini():
    '''Initialisierung

    Args:


    Returns:

    '''
    pass


def accuracy(Yte_predict, Yte):
    '''Überprüfung der richtigen Label mit den vorhergesagten

    Args:
        Yte_predict (list): vorhergesagte Labels (vom Computer)
        Yte (list): wahren Labels (vom Mensch)


    Returns:
        accuracy (list): Übereinstimmung
    '''
    return np.mean(Yte_predict == Yte)


def ratio(Yte_predict):
    '''Prozentualer Anteil aus Array

    Args:
        Yte_predict (list): vorhergesagte Labels (vom Computer)

    Returns:
        accuracy (list): relativer Anteil der Labels "1" im Array
    '''
    return np.mean(Yte_predict == 1)
