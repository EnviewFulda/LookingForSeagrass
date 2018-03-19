import cv2
import numpy as np
import argparse
from sklearn.preprocessing import binarize
import json
import os

def crop_patches (img, pos_patches_koor, neg_patches_koor, patch_height, patch_width):
    pos_patches = []
    neg_patches = []

    for i in pos_patches_koor:
        x = i[0]
        y = i[1]
        crop_img = img[y:y+patch_height, x:x+patch_width]
        pos_patches.append(crop_img)

    for i in neg_patches_koor:
        x = i[0]
        y = i[1]
        crop_img = img[y:y+patch_height, x:x+patch_width]
        neg_patches.append(crop_img)

    return pos_patches, neg_patches



def save_patches (img_path, pos_patches, neg_patches, pos_patches_save_path, neg_patches_save_path):
    base = os.path.basename(img_path)
    split = os.path.splitext(base)
    no_extension = os.path.splitext(base)[0]

    rgbimg = cv2.imread(img_path)

    pos_patch_counter = 0
    neg_patch_counter = 0

    for i in pos_patches:
        string = pos_patches_save_path + "/" + no_extension + "_" + str(pos_patch_counter) + ".jpg"
        pos_patch_counter += 1
        cv2.imwrite(string, i)

    for i in neg_patches:
        string = neg_patches_save_path + "/" + no_extension + "_" + str(neg_patch_counter) + ".jpg"
        neg_patch_counter += 1
        cv2.imwrite(string, i)


def ink_patches_in_pixelmap(img, pos_patches_koor, neg_patches_koor, patch_height, patch_width, pos_color, neg_color):
    for i in pos_patches_koor:
        x = i[0]
        y = i[1]
        cv2.rectangle(img,(x,y),(x+patch_width,y+patch_height),pos_color,-1)

    for i in neg_patches_koor:
        x = i[0]
        y = i[1]
        cv2.rectangle(img,(x,y),(x+patch_width,y+patch_height),neg_color,-1)

    return img


def simple_grid_search (img, patch_height, patch_width):
    image_height = img.shape[0]
    image_width = img.shape[1]

    y_koor = np.arange(0,image_height,patch_height)
    x_koor = np.arange(0,image_width,patch_width)
    pos_patches_koor = []
    neg_patches_koor = []

    for x in x_koor:
        for y in y_koor:
            crop_img = img[y:y+patch_height, x:x+patch_width]
            all_pixel_one = np.all(crop_img == 1)
            all_pixel_zero = np.all(crop_img == 0)

            if all_pixel_zero: pos_patches_koor.append(list(np.array([x,y])))
            if all_pixel_one: neg_patches_koor.append(list(np.array([x,y])))

    return pos_patches_koor, neg_patches_koor


def left_center_grid_search (img, patch_height, patch_width):
    image_height = img.shape[0]
    image_width = img.shape[1]

    y_koor = np.arange(0,image_height,patch_height) # Höhen unterteilen
    pos_patches_koor = []
    neg_patches_koor = []

    for y in y_koor: # iterate along height
        x = 0

        while(x <= image_width-patch_width): # iterate along width
            crop_img = img[y:y+patch_height, x:x+patch_width]
            all_pixel_one = np.all(crop_img == 1)
            all_pixel_zero = np.all(crop_img == 0)

            if all_pixel_one or all_pixel_zero:
                if all_pixel_zero: pos_patches_koor.append(list(np.array([x,y])))
                if all_pixel_one: neg_patches_koor.append(list(np.array([x,y])))
                x += patch_width
            else: x += 1

    return pos_patches_koor, neg_patches_koor


def generate_patches (rgb_image, rgb_pixelmap, patch_height, patch_width):
    #gray_pixelmap = cv2.cvtColor(rgb_pixelmap, cv2.COLOR_BGR2GRAY) # In Grauwertbild wandeln
    logical_pixelmap = binarize(rgb_pixelmap, threshold=127) # Binarisieren

    pos_patches_koor, neg_patches_koor = left_center_grid_search(logical_pixelmap, patch_height, patch_width) # Get positve/negative patches coordinates from patches search algorithm

    #img = ink_patches_in_pixelmap(rgb_pixelmap, pos_patches_koor, neg_patches_koor, int(args.patch_height), int(args.patch_width), (0,255,0), (0,0,255))
    #cv2.imshow(" ", img)
    #cv2.waitKey(0)

    return crop_patches (rgb_image, pos_patches_koor, neg_patches_koor, patch_height, patch_width) # crop postive/negative patches from rgb image based on positve/negative coordinates


if __name__ == '__main__':

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", help="annotation_set.json")
    parser.add_argument("--patch_width", help="120")
    parser.add_argument("--patch_height", help="120")
    parser.add_argument("--invert", help="0,1") # Seagras/Not Seagras
    parser.add_argument("--neg_patches", help="path")
    parser.add_argument("--pos_patches", help="path")
    args = parser.parse_args()

    patch_height = int(args.patch_height)
    patch_width = int(args.patch_width)

    pos_patches_counter = 0
    neg_patches_counter = 0
    image_counter = 0

    whole_images = len(json.load(open(args.json_file)))

    for i in json.load(open(args.json_file)):
        image_counter += 1
        print("Picture " + str(image_counter) + "/" + str(whole_images))

        for key, value in i.items():
            rgb_image = cv2.imread(key)
            rgb_pixelmap = cv2.imread(value)

            pos_patches, neg_patches = generate_patches(rgb_image, rgb_pixelmap, patch_height, patch_width)

            save_patches (key, pos_patches, neg_patches, args.pos_patches, args.neg_patches) # save positive/negative patches

            pos_patches_counter += len(pos_patches) # count positive patches
            neg_patches_counter += len(neg_patches) # count negative patches



    print ("whole pos patches:",pos_patches_counter)
    print ("whole neg patches:",neg_patches_counter)
    print ("whole pos und neg patches:",pos_patches_counter + neg_patches_counter)
