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
    '''show a picture

    Args:
        picture: image

    Returns:

    '''

    cv2.imshow(" ", picture)
    cv2.waitKey(0)


def patches(loaded_picture, coordinates, patch_size_height, patch_size_width):
    '''divide image into patches

    Args:
        picture: image

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
    '''Creates a coordinate list. Each patch at a fixed position in the image is assigned a coordinate
       The patches are iterated line by line from left to right and the coordinates are saved.

    Args:


    Returns:

    '''
    high = np.arange(0,image_size_heigh,patch_size_height) # Upper left corner
    width = np.arange(0,image_size_width,patch_size_width) # Lower right corner
    coordinates = []
    for h in high:
        for w in width:
            coordinates.append([w,h])
    return np.array(coordinates)


def logical_pixelmap(prediction, height, width, coordinates, patch_size_height,patch_size_width):
    '''Creates a logical pixelmap based on the prediction.

    Args:


    Returns:

    '''
    # logical Pixelmap:    0   seagrass                 1 background
    logical_pixelmap = np.ones(([height,width]), dtype = "uint8")

    for i in range(len(prediction)): # Go through prediction array 
        w = coordinates[i][0]
        h = coordinates[i][1]
        if prediction[i]: # 1 = Label for seagrass
            logical_pixelmap[h:h+patch_size_height, w:w+patch_size_width] = 0 # 0 = Seagrass in the logical pixelmap

    return logical_pixelmap
