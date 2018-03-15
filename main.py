import json
import cv2
from sklearn.preprocessing import binarize
import numpy as np

import argparse
from matplotlib import style
style.use("ggplot")
import argparse
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import classifier.dataset
import classifier.hog as hog
import classifier.cnn as cnn
import classifier.lbp as lbp
import classifier.rectangle as rectangle
import classifier.cal as cal
import classifier.msg as msg
import classifier.svm as svm
import classifier.csvfile as csvfile
import classifier.superpixel as superpixel
import classifier.eval_segm as eval_segm

from utils import generate_patches


from utils.generate_patches import generate_patches

import time

training_time = 0
sumarized_prediction_time = 0

SINGLE = 0
MULTIPLE = 1


dict_input = dict()
dict_output = dict()
dict_experiment = dict()
list_eval = []



def rgb_pixelmap_to_logical_pixelmap (rgb_pixelmap):
    ''' Bild in Logische Pixelmap umwandeln

    Args:

    Returns:

    '''
    gray_pixelmap = cv2.cvtColor(rgb_pixelmap, cv2.COLOR_BGR2GRAY) # In Grauwertbild wandeln
    logical_pixelmap = binarize(gray_pixelmap, threshold=127) # Binarisieren
    return logical_pixelmap

def pixelmap_to_image (pixel_map):
    '''Logische Pixelmap in Bild umwandeln

    Args:

    Returns:

    '''

    return 255 - (pixel_map * 255) # Pixelmap: 1=Seegras, 0=kein Seegras. Bild: 0=Seegras=Schwarz, 255=kein Seegras=Weiß

def ink_image(image, pixel_map):
    '''Initialisierung

    Args:
        picture: Bild, worauf die Boxen gezeichnet werden
        pixel_map: Seegras/Kein Seegras


    Returns:
        picture: Bild mit eingezeichneten Seegras

    '''

    alpha = 0.3 # Transparenz Faktor
    transparent_image = image.copy() # Transparenz
    green_image = image.copy()
    original_image = image.copy()
    return_image = image.copy()


    # Grünes Bild erstellen
    cv2.rectangle(green_image,(0,0),(1920,1080),(0,255,0),-1)

    # Originalbild mit Grün überlagern
    cv2.addWeighted(green_image, alpha, original_image, 1 - alpha, 0, transparent_image) # Transparenz


    image = np.array(image)

    height = image.shape[0]
    width  = image.shape[1]

    pm_height = pixel_map.shape[0]
    pm_width  = pixel_map.shape[1]

    #print("Bild", height,width)
    #print("PM", pm_height,pm_width)

    for h in range (height):
        for w in range (width):
            if pixel_map[h][w] == 0: # 0 = Schwarz = Seegras
                return_image[h][w] = transparent_image[h][w]

    return return_image




def generate_training_data(root_path, json_path, patch_size_height, patch_size_width, depth_min, depth_max, show=False):
    list_patches = []
    list_labels = []

    for i in json.load(open(root_path + "/" + json_path)):
        """
        print(i["coverage"])
        print(i["depth"])
        print(i["ground-truth"])
        print(i["image"])
        print("---")
        """

        if check_depth (i["depth"], depth_min, depth_max):

            path_rgb_image = os.path.join(root_path,i["image"])

            rgb_image = cv2.imread(path_rgb_image)
            if rgb_image is not None:
                if show:
                    cv2.imshow(" ", rgb_image)
                    cv2.waitKey(0)

                path_rgb_pixelmap = os.path.join(root_path,i["ground-truth"])
                rgb_pixelmap = cv2.imread(path_rgb_pixelmap)
                if rgb_pixelmap is not None:
                    pos_patches, neg_patches = generate_patches (rgb_image, rgb_pixelmap, patch_size_height, patch_size_width)
                    # gehe in die Listen rein
                    for i in pos_patches:
                        list_patches.append(i) # hole positiven Patch
                        list_labels.append(1) # setze Label dieses Patches auf 1

                    for i in neg_patches:
                        list_patches.append(i) # hole negativen Patch
                        list_labels.append(0) # setze Label dieses Patches auf 0
                else:
                    msg.timemsg('Could not load pixelmap: {}'.format(path_rgb_pixelmap))
            else:
                msg.timemsg('Could not load image: {}'.format(path_rgb_image))

    return list_patches, list_labels



def train(root_path, json_path, features, patch_size_height, patch_size_width, graph, depth_min, depth_max, args):

    # Zeitmessung
    start_time = time.time()


    # generate patches
    msg.timemsg("Generiere Patches start")
    list_patches, list_labels = generate_training_data(root_path, json_path, patch_size_height, patch_size_width, depth_min, depth_max)
    msg.timemsg("Generiere Patches fertig")

    msg.timemsg(str(len(list_patches)) + " Patches generiert")

    # init svm
    svm.ini() # SVM initalisieren

    # extract features by respective method
    msg.timemsg("Generiere Features start")
    base_path = os.path.split(args.output)[0]
    path = os.path.join(base_path, get_dumpname(args))
    path += '.train.feat'
    X = get_features(features, list_patches, path)
    # if (features == "hog"): features = hog.features(list_patches, MULTIPLE)
    # if (features == "cnn"): features = cnn.features(list_patches, MULTIPLE)
    # if (features == "lbp"): features = lbp.features(list_patches, MULTIPLE)
    msg.timemsg("Generiere Features fertig")

    # train svm
    msg.timemsg("Trainiere SVM start")
    svm.train(np.array(X),np.array(list_labels)) # SVM trainieren
    msg.timemsg("Trainiere SVM stop")

    # Zeitmessung
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

    for i in json.load(open(root_path + "/" + json_path)):
        if check_depth (i["depth"], depth_min, depth_max):

            # Zeitmessung
            start_time = time.time()

            path_rgb_image = os.path.join(root_path,i["image"])
            loaded_picture = cv2.imread(path_rgb_image)
            path_rgb_pixelmap = os.path.join(root_path,i["ground-truth"])
            loaded_pixelmap = cv2.imread(path_rgb_pixelmap)
            if loaded_picture is not None and loaded_pixelmap is not None:
                height = loaded_picture.shape[0]
                width  = loaded_picture.shape[1]

                dumpname = get_dumpname(args)
                outdir = os.path.join(os.path.split(args.output)[0], dumpname)
                #if not os.path.exists(outdir):
                #    os.mkdir(outdir)
                base_path = os.path.join(outdir, '{}.predict'.format(os.path.basename(path_rgb_image)))
                feat_path = base_path + '.feat'
                segment_path = base_path + '.segment'
                # patches = None
                # if not os.path.exists(path):
                # three cases: RP, SP_SLIC, SP_CW
                # if SP_SLIC or SP_CW -> generate rect patches from superpixel

                # if os.path.exists(segment_path):
                if False:
                    with open( segment_path, "rb" ) as f:
                        segments = pickle.load(f)
                        patches = None
                else:
                    if pattern == "RP": # rectangle pattern
                        coordinates = rectangle.create_coordinates_list (patch_size_height, patch_size_width, height, width)
                        patches = rectangle.patches(loaded_picture, coordinates, patch_size_height, patch_size_width)

                    if pattern == "SP_SLIC": # simple linear iterative clustering
                        patches, segments = superpixel.patches(loaded_picture, "SP_SLIC", patch_size_height, patch_size_width, height, width) # Patches wurden in Rechtecke umgewandelt, Segmente werden als Koordinaten beschrieben

                    if pattern == "SP_CW": # compact watershed
                        patches, segments = superpixel.patches(loaded_picture, "SP_CW", patch_size_height, patch_size_width, height, width) # Patches wurden in Rechtecke umgewandelt, Segmente werden als Koordinaten beschrieben
                    # if "SP" in pattern:
                    #     with open(segment_path, "wb") as f:
                    #         pickle.dump(segments, f)

                # if pattern == "RP": # rectangle pattern
                #     coordinates = rectangle.create_coordinates_list (patch_size_height, patch_size_width, height, width)
                #     patches = rectangle.patches(loaded_picture, coordinates, patch_size_height, patch_size_width)
                #
                # if pattern == "SP_SLIC": # simple linear iterative clustering
                #     patches, segments = superpixel.patches(loaded_picture, "SP_SLIC", patch_size_height, patch_size_width, height, width) # Patches wurden in Rechtecke umgewandelt, Segmente werden als Koordinaten beschrieben
                #
                # if pattern == "SP_CW": # compact watershed
                #     patches, segments = superpixel.patches(loaded_picture, "SP_CW", patch_size_height, patch_size_width, height, width) # Patches wurden in Rechtecke umgewandelt, Segmente werden als Koordinaten beschrieben


                # feature extraction: LBP, HOG, CNN
                X = get_features(features, patches, feat_path)
                # if (args.features == "hog"): features = hog.features(patches, MULTIPLE)
                # if (args.features == "cnn"): features = cnn.features(patches, MULTIPLE)
                # if (args.features == "lbp"): features = lbp.features(patches, MULTIPLE)


                # prediction
                prediction = svm.predict(X, MULTIPLE)


                # Zeitmessung
                elapsed_time = (time.time() - start_time) * 1000 # ms
                global sumarized_prediction_time
                sumarized_prediction_time += elapsed_time



                # calculate pixelmaps
                if pattern == "RP":
                    coordinates = rectangle.create_coordinates_list (patch_size_height, patch_size_width, height, width)
                    classifier_pixel_map = rectangle.logical_pixelmap(prediction, height, width, coordinates, patch_size_height, patch_size_width)

                if pattern == "SP_SLIC" or pattern == "SP_CW":
                    classifier_pixel_map = superpixel.logical_pixelmap(segments, prediction, height, width) # In Segments stecken die Koordinaten jedes Pixels für das Segment


                # Evaluation
                annotated_logical_pixelmap = rgb_pixelmap_to_logical_pixelmap(loaded_pixelmap)

                dict_bundle = evaluate (i["image"], classifier_pixel_map, annotated_logical_pixelmap)

                global list_eval
                list_eval.append(dict_bundle)



                # Debug
                if (args.mode == "debug"):
                    # Anzeigen des Bildes mit eingezeichnetem Seegras
                    picture = ink_image(loaded_picture, classifier_pixel_map)
                    if show:
                        cv2.imshow(" ", picture)
                        cv2.waitKey(0)
                    if write:
                        cv2.imwrite('debug_inked_image.jpg', picture)
                        input('See written images at: {}'.format(os.getcwd()))
            else:
                if loaded_pixelmap is None:
                    msg.timemsg('Could not load pixelmap: {}'.format(path_rgb_pixelmap))
                if loaded_picture is None:
                    msg.timemsg('Could not load image: {}'.format(path_rgb_image))




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
    if len(list): # Wenn Liste nicht leer ist
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

def get_features(feature_method, patches, path):
    # if os.path.exists(path):
    if False:
        msg.timemsg("Load {} feature dump from {}".format(feature_method, path))
        with open( path, "rb" ) as f:
            X = pickle.load(f)
        msg.timemsg("Load featue dump finished")
    else:
        msg.timemsg("Generiere {} Features start".format(feature_method))
        if (feature_method == "hog"): X = hog.features(patches, MULTIPLE)
        if (feature_method == "cnn"): X = cnn.features(patches, MULTIPLE)
        if (feature_method == "lbp"): X = lbp.features(patches, MULTIPLE)
        msg.timemsg("Generiere Features fertig")
        # msg.timemsg("Dump features to {}".format(path))
        # with open(path, "wb") as f:
        #     pickle.dump(X, f)
        # msg.timemsg("Feature dump fertig")
    return X



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", help="RP, SP_SLIC, SP_CW")
    parser.add_argument("--features", help="hoc, lbp, cnn")
    parser.add_argument("--graph", help="/path/to/classify_image_graph_def.pb")
    parser.add_argument("--mode", help="preview, eval")

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

    """
    train()
    for x in test.json:
        predict()
        result.append(evaluate())

    return result
    """

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
    msg.timemsg("Training gestartet")
    train(args.folder_root, args.eval_train, args.features, int(args.patch_size_height), int(args.patch_size_width), args.graph, float(args.depth_min), float(args.depth_max), args)
    dict_experiment["training"] = training_time
    msg.timemsg("Training abgeschlossen")



    msg.timemsg("Prediction gestartet")
    prediction(args.folder_root, args.eval_test, args.pattern, args.features, int(args.patch_size_height), int(args.patch_size_width), float(args.depth_min), float(args.depth_max), args)
    dict_experiment["prediction"] = sumarized_prediction_time
    msg.timemsg("Prediction abgeschlossen")


    # Output Json erzeugen
    create_output_dict(args.output)
    msg.timemsg("eval json erzeugt")
