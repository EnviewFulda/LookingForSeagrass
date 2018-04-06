#!/usr/bin/env python

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2


from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import watershed


with_neightbor = 1


def ini():
    '''initialization

    Args:


    Returns:

    '''





def patches(Picture, method, patch_size_height, patch_size_width, height, width):
    '''Dividing an image into superpixels (patches)

    Args:
        picture: image

    Returns:

    '''


    segments_height = int (height / patch_size_height)
    segments_width = int (width / patch_size_width)
    number_of_segments = int (segments_height * segments_width)

    listPatches = []

    if method == "SP_SLIC":
        segments = slic(img_as_float(Picture), n_segments = number_of_segments, sigma = 5)

    if method == "SP_CW":
        gradient = sobel(rgb2gray(Picture))
        segments = watershed(gradient, markers=number_of_segments, compactness=0.0001)

    #pic.show(mark_boundaries(Picture, segments, (255,255,255))) # show segments


    original_image = np.copy(Picture)

    for i in np.unique(segments): # number of segments


        # draw mask
        image_mask = np.zeros(Picture.shape[:2], dtype = "uint8") # 3D to 1D
        w = np.where(segments == i)
        image_mask[w] = 255

        x,y,w,h = cv2.boundingRect(image_mask) # calculate rectangle

        segment_mask_in_original_image = cv2.bitwise_and(original_image,original_image,mask = image_mask)

        if (with_neightbor):
            # with neighborhood
            segment_cropped_with_neightbor = original_image[y:y+h,x:x+w]
            segment_cropped_with_neightbor_and_reqized = cv2.resize(segment_cropped_with_neightbor, (patch_size_width, patch_size_height)) # width, height

            listPatches.append(segment_cropped_with_neightbor_and_reqized)
        else:
            # without neighborhood
            segment_cropped_without_neightbor = segment_mask_in_original_image[y:y+h,x:x+w]
            segment_cropped_without_neightbor_and_reqized = cv2.resize(segment_cropped_without_neightbor, (patch_size_width, patch_size_height)) # width, height

            listPatches.append(segment_cropped_without_neightbor_and_reqized)


    return listPatches, segments



def logical_pixelmap(segments, prediction, height, width):
    '''Creates a logical pixelmap based on the prediction.

    Args:


    Returns:

    '''
    # logical Pixelmap:    0   seagrass                 1 background
    logical_pixelmap = np.ones(([height,width]), dtype = "uint8")

    for i in range(len(prediction)): # go through prediction array
        if prediction[i]: # 1 = label for seagrass
            w = np.where(segments == i)
            logical_pixelmap[w] = 0 # 0 = seagrass in the logical pixel map
    return logical_pixelmap
