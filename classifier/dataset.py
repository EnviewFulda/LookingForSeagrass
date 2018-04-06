#!/usr/bin/env python

import json

def ini():
    '''Initialisierung

    Args:


    Returns:

    '''
    pass


def labels(json_file):
    '''Liefert die Labels

    Args:
        json_file (dict): Pfade und Labels

    Returns:
        labels (list)
    '''
    labels = []
    for i in json.load(open(json_file)):
        for key, value in i.items():
            labels.append(int(value)) # Casten auf Int
    return labels


def path(json_file):
    '''Liefert die Pfade

    Args:
        json_file (dict): Pfade und Labels

    Returns:
        pfade (list)
    '''
    path = []
    for i in json.load(open(json_file)):
        for key, value in i.items():
            path.append(key)
    return path
