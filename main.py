#!/usr/bin/env python

import json
import cv2
import traceback
from sklearn.preprocessing import binarize
import numpy as np

import argparse
from matplotlib import style
style.use("ggplot")
import argparse
import os
import pickle
import classifier.dataset
import classifier.hog as hog
import classifier.cnn as cnn
import classifier.lbp as lbp
import classifier.rectangle as rectangle
import classifier.cal as cal
import classifier.msg as msg
import classifier.lr as lr
import classifier.csvfile as csvfile
import classifier.superpixel as superpixel
import classifier.eval_segm as eval_segm
from skimage.io import imread

from utils import generate_patches
from utils.generate_patches import generate_patches

import time

training_time = 0
sumarized_prediction_time = 0

SINGLE = 0
MULTIPLE = 1
SERIALIZE_FEATURES = True
BATCH_SIZE = 1000
dict_input = dict()
dict_output = dict()
dict_experiment = dict()
list_eval = []

def rgb_pixelmap_to_logical_pixelmap (rgb_pixelmap):
    ''' Turn image into logical binary pixel map.

    Args:

    Returns:

    '''
    gray_pixelmap = rgb_pixelmap
    logical_pixelmap = binarize(gray_pixelmap, threshold=127) # Binarisieren
    return logical_pixelmap

def pixelmap_to_image (pixel_map):
    '''transform binary pixel map to image (rgb)

    Args:

    Returns:

    '''

    return 255 - (pixel_map * 255) # Pixelmap: 1=seagrass, 0=background. image: 0=seagrass=black, 255=background=white

def ink_image(image, pixel_map):
    '''

    Args:
        picture: image to be colored
        pixel_map: seagrass / background


    Returns:
        picture: image with colored regions of seagrass

    '''

    alpha = 0.3 # Transparenz Faktor
    transparent_image = image.copy() # Transparenz
    green_image = image.copy()
    original_image = image.copy()
    return_image = image.copy()


    # create green image
    cv2.rectangle(green_image,(0,0),(1920,1080),(0,255,0),-1)

    # overlay original image with green image
    cv2.addWeighted(green_image, alpha, original_image, 1 - alpha, 0, transparent_image) 


    image = np.array(image)

    height = image.shape[0]
    width  = image.shape[1]

    for h in range (height):
        for w in range (width):
            if pixel_map[h][w] == 0: # 0=black=seagrass
                return_image[h][w] = transparent_image[h][w]

    return return_image

def generate_training_data(root_path, json_path, patch_size_height, patch_size_width, depth_min, depth_max, show=False, batch_size=2000):
    list_patches = []
    list_labels = []
    counter = 1

    for i in json.load(open(root_path + "/" + json_path)):
        if check_depth (i["depth"], depth_min, depth_max):

            path_rgb_image = os.path.join(root_path,i["image"])

            try:
                rgb_image = imread(path_rgb_image)
            except:
                msg.timemsg('Could not load image: {}'.format(path_rgb_image))
                continue
            if rgb_image is not None:
                if show:
                    cv2.imshow(" ", rgb_image)
                    cv2.waitKey(0)
                path_rgb_pixelmap = os.path.join(root_path,i["ground-truth"])
                try:
                    rgb_pixelmap = imread(path_rgb_pixelmap)
                except:
                    msg.timemsg('Could not load pixelmap: {}'.format(path_rgb_pixelmap))
                    continue
                if rgb_pixelmap is not None:
                    pos_patches, neg_patches = generate_patches (rgb_image, rgb_pixelmap, patch_size_height, patch_size_width)
                    # gehe in die Listen rein
                    for i in pos_patches:
                        list_patches.append(i) # get positive patch
                        list_labels.append(1) # set label of this patch to one

                    for i in neg_patches:
                        list_patches.append(i) # get negative patch
                        list_labels.append(0) # set label of this patch to zero
                    if counter % batch_size == 0:
                        yield list_patches, list_labels
                        list_patches = []
                        list_labels = []
                    counter += 1
                else:
                    msg.timemsg('Could not load pixelmap: {}'.format(path_rgb_pixelmap))
            else:
                msg.timemsg('Could not load image: {}'.format(path_rgb_image))
    yield list_patches, list_labels

def train(root_path, json_path, features, patch_size_height, patch_size_width, graph, depth_min, depth_max, args):

    # Zeitmessung
    start_time = time.time()

    counter = 0 
    labels = []
    # generate patches
    msg.timemsg("Generate patches start")
    with open(root_path + "/" + json_path) as f:
        train_list = json.load(f)

    base_path = os.path.split(args.output)[0]
    path = os.path.join(base_path, get_dumpname(args))
    clf_path = path + '.clf'
    if os.path.exists(clf_path):
        lr.ini(path=clf_path)
    else:
        for list_patches, list_labels in generate_training_data(root_path, json_path, patch_size_height, 
                                        patch_size_width, depth_min, depth_max, batch_size=BATCH_SIZE):
            msg.timemsg("Batch {}: Patch generation done".format(counter))
            msg.timemsg(str(len(list_patches)) + "Patches have been generated")
            # extract features by respective method
            msg.timemsg("Batch {}: Generate features start".format(counter))
            feat_path = path + '.train.batch{}.feat'.format(counter)
            X_split = get_features(features, list_patches, feat_path, serialize=SERIALIZE_FEATURES)
            msg.timemsg("Batch {}: Generate features done".format(counter))
            if counter == 0:
                X = X_split
            else:
                X = np.vstack((X, X_split))
            labels += list_labels
            counter += 1
        msg.timemsg('Generated all features!')

        lbl_path = path + '.train.lbls'
        if os.path.exists(lbl_path):
            labels = __load(lbl_path)
        else:
            __dump(labels, lbl_path)
        # init lr
        lr.ini() 
        # train lr
        msg.timemsg("Training Classifier start")
        lr.train(np.array(X),np.array(labels), path=clf_path) 
        msg.timemsg("Training Classifier end")

        # time measurement
        elapsed_time = (time.time() - start_time) * 1000 # ms
        global training_time
        training_time = elapsed_time

def check_depth (depth, depth_min, depth_max):
    if depth >= depth_min and depth < depth_max:
        return 1
    else:
        return 0

def get_dumpname(args):
    dumpname = "{}_{}_{}X{}_{}_{}".format(args.pattern, args.features,
        args.patch_size_width, args.patch_size_height,
        args.depth_min, args.depth_max)
    return dumpname

def prediction(root_path, json_path, pattern, features, patch_size_height, patch_size_width, depth_min, depth_max, args, show=False, write=True):
    test_json = json.load(open(root_path + "/" + json_path))
    n_test = len(test_json)
    counter = 1
    for i in test_json:
        if check_depth (i["depth"], depth_min, depth_max):
            # Zeitmessung
            start_time = time.time()
            path_rgb_image = os.path.join(root_path,i["image"])
            try:
                loaded_picture = imread(path_rgb_image)
            except:
                msg.timemsg('Could not load image: {}'.format(path_rgb_image))
                continue
            path_rgb_pixelmap = os.path.join(root_path,i["ground-truth"])
            try:
                loaded_pixelmap = imread(path_rgb_pixelmap)
            except:
                msg.timemsg('Could not load pixelmap: {}'.format(path_rgb_pixelmap))
                continue
            if loaded_picture is not None and loaded_pixelmap is not None:
                height = loaded_picture.shape[0]
                width  = loaded_picture.shape[1]
                dumpname = get_dumpname(args)
                if pattern == "RP": # rectangle pattern
                    coordinates = rectangle.create_coordinates_list (patch_size_height, patch_size_width, height, width)
                    patches = rectangle.patches(loaded_picture, coordinates, patch_size_height, patch_size_width)
                if pattern == "SP_SLIC": # simple linear iterative clustering
                    patches, segments = superpixel.patches(loaded_picture, "SP_SLIC", patch_size_height, patch_size_width, height, width) # Patches wurden in Rechtecke umgewandelt, Segmente werden als Koordinaten beschrieben
                if pattern == "SP_CW": # compact watershed
                    patches, segments = superpixel.patches(loaded_picture, "SP_CW", patch_size_height, patch_size_width, height, width) # Patches wurden in Rechtecke umgewandelt, Segmente werden als Koordinaten beschrieben
                # feature extraction: LBP, HOG, CNN
                X = get_features(features, patches, '', serialize=False)
                # prediction
                prediction = lr.predict(X, MULTIPLE)
                # time measurement
                elapsed_time = (time.time() - start_time) * 1000 # ms
                global sumarized_prediction_time
                sumarized_prediction_time += elapsed_time
                # calculate pixelmaps
                if pattern == "RP":
                    coordinates = rectangle.create_coordinates_list (patch_size_height, patch_size_width, height, width)
                    classifier_pixel_map = rectangle.logical_pixelmap(prediction, height, width, coordinates, patch_size_height, patch_size_width)
                if pattern == "SP_SLIC" or pattern == "SP_CW":
                    classifier_pixel_map = superpixel.logical_pixelmap(segments, prediction, height, width) # In Segments stecken die Koordinaten jedes Pixels für das Segment
                # evaluation
                annotated_logical_pixelmap = rgb_pixelmap_to_logical_pixelmap(loaded_pixelmap)
                dict_bundle = evaluate (i["image"], classifier_pixel_map, annotated_logical_pixelmap)
                global list_eval
                list_eval.append(dict_bundle)
                # debug
                if (args.mode == "debug"):
                    # show colored picture
                    picture = ink_image(loaded_picture, classifier_pixel_map)
                    if show:
                        cv2.imshow(" ", picture)
                        cv2.waitKey(0)
                    if write:
                        debug_base_path = os.path.split(args.output)[0]
                        debug_img_name = os.path.basename(path_rgb_image)
                        debug_out_path = os.path.join(debug_base_path, dumpname + debug_img_name )
                        cv2.imwrite(debug_out_path, cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
            else:
                if loaded_pixelmap is None:
                    msg.timemsg('Could not load pixelmap: {}'.format(path_rgb_pixelmap))
                if loaded_picture is None:
                    msg.timemsg('Could not load image: {}'.format(path_rgb_image))
            msg.timemsg('Prediction Progress: {}%'.format(float(counter/n_test)*100))
            counter += 1

def evaluate(path, pm_clf, pm_ann):

    # eval pixel map + prediction pixel map
    # four methods.

    dict_bundle = dict()
    dict_bundle["img-path"] = path
    dict_bundle["meanIU"] = str(eval_segm.mean_IU(pm_clf, pm_ann)) # eval_segm, gt_segm
    dict_bundle["fwIU"] =  str(eval_segm.frequency_weighted_IU(pm_clf, pm_ann))
    dict_bundle["pixel-acc"] = str(eval_segm.pixel_accuracy(pm_clf,pm_ann))
    dict_bundle["mean-acc"] = str(eval_segm.mean_accuracy(pm_clf, pm_ann))

    return dict_bundle

def save_json(path, list):
    if len(list): 
        your_json = json.dumps(list)
        parsed = json.loads(your_json)
        a = json.dumps(parsed, indent=4, sort_keys=True)
        file = open(path,"w")
        file.write(a)
        file.close()

def create_output_dict (path):

    dict_eval_file = dict()

    dict_output["exp_time"] = dict_experiment
    dict_output["eval"] = list_eval

    dict_eval_file["Input"] = dict_input
    dict_eval_file["Output"] = dict_output

    save_json(path, dict_eval_file)

def get_feat(feature_method, patches):
    msg.timemsg("Generiere {} Features start".format(feature_method))
    if (feature_method == "hog"): X = hog.features(patches, MULTIPLE)
    if (feature_method == "cnn"): X = cnn.cnn_instance.features(patches, MULTIPLE)
    if (feature_method == "lbp"): X = lbp.features(patches, MULTIPLE)
    msg.timemsg("Generiere Features fertig")
    return X

def __load(path):
    if os.path.exists(path):
        msg.timemsg("Load dump from {}".format(path))
        with open( path, "rb" ) as f:
            X = pickle.load(f)
        msg.timemsg("Load dump done")
        return X
    else:
        msg.timemsg('Could not load pickle: {}'.format(path))
        return None

def __dump(X, path):
    if os.path.exists(path):
        msg.timemsg('Will overite file {} with new pickle dump'.format(path))
    else:
        try:
            msg.timemsg("Dump to {}".format(path))
            with open(path, "wb") as f:
                pickle.dump(X, f, protocol=pickle.HIGHEST_PROTOCOL)
            msg.timemsg("Dump done")
        except:
            msg.timemsg('Exception during dump of {}'.format(path))
            msg.timemsg(traceback.format_exec())

def get_features(feature_method, patches, path, serialize=True, chunck_size=100000):
    if serialize:
        if os.path.exists(path):
            X = __load(path)
        else:
            X = get_feat(feature_method, patches)
            __dump(X, path)
    else:
        X = get_feat(feature_method, patches)

    return np.array(X)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", help="RP, SP_SLIC, SP_CW")
    parser.add_argument("--features", help="hog, lbp, cnn")
    parser.add_argument("--graph", help="/path/to/classify_image_graph_def.pb")
    parser.add_argument("--mode", help="debug, dontcare")

    parser.add_argument("--folder_root", help="")
    parser.add_argument("--folder_images", help="")
    parser.add_argument("--folder_ground_truth", help="")
    parser.add_argument("--eval_test", help="")
    parser.add_argument("--eval_train", help="")
    parser.add_argument("--eval_validate", help="")

    parser.add_argument("--patch_size_width", help="")
    parser.add_argument("--patch_size_height", help="")

    parser.add_argument("--output", help="path/to/json/output.json")

    parser.add_argument("--depth_min", help="")
    parser.add_argument("--depth_max", help="")

    args = parser.parse_args()

    dict_input["depth_min"] = args.depth_min
    dict_input["depth_max"] = args.depth_max
    dict_input["patch_size_width"] = args.patch_size_width
    dict_input["patch_size_height"] = args.patch_size_height
    dict_input["feature"] = args.features
    dict_input["patch_type"] = args.pattern

    msg.ini(args.output + '.log')
    # init respective feature extraction method
    if (args.features == "hog"): hog.ini()
    if (args.features == "cnn"): cnn.ini(args.graph)
    if (args.features == "lbp"): lbp.ini()
    msg.timemsg("Training started")
    train(args.folder_root, args.eval_train, args.features, int(args.patch_size_height), int(args.patch_size_width), args.graph, float(args.depth_min), float(args.depth_max), args)
    dict_experiment["training"] = training_time
    msg.timemsg("Training finished")

    msg.timemsg("Prediction started")
    prediction(args.folder_root, args.eval_test, args.pattern, args.features, int(args.patch_size_height), int(args.patch_size_width), float(args.depth_min), float(args.depth_max), args)
    dict_experiment["prediction"] = sumarized_prediction_time
    msg.timemsg("Prediction finished")

    if (args.features == "cnn"): cnn.close()

    # generate json output
    create_output_dict(args.output)
    msg.timemsg("generated eval json")
