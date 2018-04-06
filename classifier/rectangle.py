#!/usr/bin/env python

import cv2
import numpy as np

def ini():
    '''initialization

    Args:


    Returns:

    '''
    pass

def show(picture):
    '''Zeit ein Bild an

    Args:
        picture: Bild

    Returns:

    '''

    cv2.imshow(" ", picture)
    cv2.waitKey(0)


def patches(loaded_picture, coordinates, patch_size_height, patch_size_width):
    '''Bild zerlegen in Patches

    Args:
        picture: Bild

    Returns:
        patches (list)
    '''

    listPatches = []

    for c in coordinates:
        w = c[0]
        h = c[1]
        crop_img = loaded_picture[h:h+patch_size_height, w:w+patch_size_width]
        listPatches.append(crop_img)

    return listPatches


def create_coordinates_list (patch_size_height, patch_size_width, image_size_heigh, image_size_width):
    '''Erstellt eine Koordinaten Liste. Jeder Patch an einer festen Stelle im Bild bekommt eine Koordinate zugeordnet
       Die Patches werden zeilenweise von links nach rechts iteriert und die Koordinaten gespeichert.

    Args:


    Returns:

    '''
    high = np.arange(0,image_size_heigh,patch_size_height) # Obere Linke Ecke
    width = np.arange(0,image_size_width,patch_size_width) # Untere Rechte Ecke
    coordinates = []
    for h in high:
        for w in width:
            coordinates.append([w,h])
    return np.array(coordinates)


def logical_pixelmap(prediction, height, width, coordinates, patch_size_height,patch_size_width):
    '''Erstellt anhand der Prediction eine logische Pixelmap.

    Args:


    Returns:

    '''
    # logical Pixelmap:    0   Seegras                 1 kein Seegras
    logical_pixelmap = np.ones(([height,width]), dtype = "uint8")

    for i in range(len(prediction)): # Gehe Array Prediction durch
        w = coordinates[i][0]
        h = coordinates[i][1]
        if prediction[i]: # 1 = Label für Seegras
            logical_pixelmap[h:h+patch_size_height, w:w+patch_size_width] = 0 # 0 = Seegras in der logischen Pixelmap

    return logical_pixelmap
