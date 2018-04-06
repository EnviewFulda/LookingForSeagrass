#!/usr/bin/env python

import json

def ini():
    '''initialization

    Args:


    Returns:

    '''
    pass


def labels(json_file):
    '''get labels

    Args:
        json_file (dict): paths and labels

    Returns:
        labels (list)
    '''
    labels = []
    for i in json.load(open(json_file)):
        for key, value in i.items():
            labels.append(int(value)) # cast to int
    return labels


def path(json_file):
    '''Liefert die Pfade

    Args:
        json_file (dict): paths and labels

    Returns:
        pfade (list)
    '''
    path = []
    for i in json.load(open(json_file)):
        for key, value in i.items():
            path.append(key)
    return path
