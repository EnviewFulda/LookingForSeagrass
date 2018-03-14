import os
import fnmatch
import argparse
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import shutil
import cv2
import json
import shutil
import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
import sys

# In der CSV Datei die Spalten Indizes
FILE_PICTURE = 1
DEPTH4_INDEX = 8

# Klassen
aClass_Low = np.arange(0.0, 100.0, 1.0)
aClass_High= np.arange(1.0, 101.0, 1.0)

def create_random_index_lists (length, percentage_train_set, percentage_test_set):

    train_len = int (length * float(percentage_train_set))
    test_len = int (length * float(percentage_test_set))

    x = np.arange(length) # Array mit Zahlen von 0 bis Anzahl der Elemente

    np.random.shuffle(x) # Verwüfeln

    train_set_rndnum = x[0:train_len]
    x = np.setdiff1d(x,train_set_rndnum) # train set aus Liste löschen

    np.random.shuffle(x) # Verwüfeln, weil setdiff1d aufsteigend sortiert

    test_set_rndnum = x[0:test_len]
    x = np.setdiff1d(x,test_set_rndnum) # test set aus Liste löschen

    np.random.shuffle(x) # Verwüfeln, weil setdiff1d aufsteigend sortiert

    validate_set_rndnum = x # validate set besteht aus dem Rest

    return train_set_rndnum, test_set_rndnum, validate_set_rndnum



def class_dict_to_datasets (class_dict, train_percentage, test_percentage):
    list_train = []
    list_test = []
    list_validate = []
    list_all = []

    print("-----")

    for c in range(len(class_dict)): # c = class (for example class 0.0-1.0)

        len_c = len(class_dict[c]) # len of class

        #print(c, len_c)

        train_set_rndnum, test_set_rndnum, validate_set_rndnum = create_random_index_lists (len_c, train_percentage, test_percentage)

        for i in train_set_rndnum:
            list_train.append(class_dict[c][i])
            list_all.append(class_dict[c][i])

        for i in test_set_rndnum:
            list_test.append(class_dict[c][i])
            list_all.append(class_dict[c][i])

        for i in validate_set_rndnum:
            list_validate.append(class_dict[c][i])
            list_all.append(class_dict[c][i])

        # debug
        if len(train_set_rndnum):
            # Aus dirname einen Ordner rauspicken
            dirname = os.path.dirname(class_dict[c][i]["image"])
            split_dirname = dirname.split("/")
            ordner_name = split_dirname[-1]
            print(ordner_name + " Train-Set    " + str(len(train_set_rndnum)))

        if len(test_set_rndnum):
            # Aus dirname einen Ordner rauspicken
            dirname = os.path.dirname(class_dict[c][i]["image"])
            split_dirname = dirname.split("/")
            ordner_name = split_dirname[-1]
            print(ordner_name + " Test-Set     " + str(len(test_set_rndnum)))

        if len(validate_set_rndnum):
            # Aus dirname einen Ordner rauspicken
            dirname = os.path.dirname(class_dict[c][i]["image"])
            split_dirname = dirname.split("/")
            ordner_name = split_dirname[-1]
            print(ordner_name + " Validate-Set " + str(len(validate_set_rndnum)))


        print("-----")

    return list_train, list_test, list_validate, list_all


def save_json(path, list):
    if len(list): # Wenn Liste nicht leer ist
        your_json = json.dumps(list)
        parsed = json.loads(your_json)
        a = json.dumps(parsed, indent=4, sort_keys=True)
        file = open(path,"w")
        file.write(a)
        file.close()


def create_notannotated_dict(dict_notannotated_path, df):

    list_notannotated = []
    # Bildersuche
    for i in df.values:
        picture_name = i[FILE_PICTURE] # Hole Bildnamen aus der CSV Datei
        try:
            path_image = dict_notannotated_path[picture_name] # Versuche Bildnamen in dict "images" zu finden

            depth4 = i[DEPTH4_INDEX]

            for a in range(len(aClass_Low)): # Klassen iterieren
                if depth4 >= aClass_Low[a] and depth4 < aClass_High[a]:

                    dict_bundle = dict()
                    dict_bundle["image"] = os.path.relpath(path_image,args.folder_root) # remove
                    dict_bundle["ground-truth"] = ""
                    dict_bundle["depth"] = depth4
                    dict_bundle["coverage"] = ""

                    list_notannotated.append(dict_bundle)
        except KeyError:
            pass

    return list_notannotated


def create_class_dict(dict_images_path, dict_groundtruth_path, df):

    matching_counter = 0

    class_dict = {}
    # Bildersuche
    for i in df.values:
        picture_name = i[FILE_PICTURE] # Hole Bildnamen aus der CSV Datei
        try:
            path_image = dict_images_path[picture_name] # Versuche Bildnamen in dict "images" zu finden
            try:
                path_groundtruth = dict_groundtruth_path["pm_" + picture_name] # Verscuhe "pm_" + Bildnamen in dict "ground-truth" zu finden
                matching_counter += 1

                depth4 = i[DEPTH4_INDEX]

                for a in range(len(aClass_Low)): # Klassen iterieren
                    if depth4 >= aClass_Low[a] and depth4 < aClass_High[a]:

                        dict_bundle = dict()
                        dict_bundle["image"] = os.path.relpath(path_image,args.folder_root) # remove
                        dict_bundle["ground-truth"] = os.path.relpath(path_groundtruth,args.folder_root) # remove
                        dict_bundle["depth"] = depth4
                        dict_bundle["coverage"] = str(coverage_seagras_in_pixelmap(path_groundtruth))

                        if a not in class_dict:
                            class_dict[a] = list()
                        class_dict[a].append(dict_bundle)

            except KeyError:
                pass
        except KeyError:
            pass


    print("Match in folder (ground-truth, image) and file (allrovcomb5.csv): " + str(matching_counter))

    return class_dict



def image_path_to_dict (path):
    # Suche nach .jpg Bildern und speicher sie in einem dictionary
    dict_images_path = dict()
    for path, dirs, files in os.walk(path):
        for f in files:
            if fnmatch.fnmatch(f, '*.jpg'):
                dict_images_path[f] = os.path.join(path, f) # key = Bildname
    return dict_images_path


def coverage_seagras_in_pixelmap (path):
    rgb_pixelmap = cv2.imread(path)
    gray_pixelmap = cv2.cvtColor(rgb_pixelmap, cv2.COLOR_BGR2GRAY) # In Grauwertbild wandeln
    logical_pixelmap = binarize(gray_pixelmap, threshold=127) # Binarisieren

    return 1.0 - np.mean(logical_pixelmap)




# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("--folder_root", help="")
parser.add_argument("--folder_images", help="")
parser.add_argument("--folder_ground_truth", help="")
parser.add_argument("--folder_not_annotated", help="")
parser.add_argument("--test_json", help="")
parser.add_argument("--train_json", help="")
parser.add_argument("--validate_json", help="")
parser.add_argument("--test_percentage", help="")
parser.add_argument("--train_percentage", help="")
parser.add_argument("--validate_percentage", help="")
parser.add_argument("--all_json", help="")
parser.add_argument("--allrovcomb5_csv", help="")
args = parser.parse_args()




dict_images_path = image_path_to_dict(args.folder_root + "/" + args.folder_images)
dict_groundtruth_path = image_path_to_dict(args.folder_root + "/" + args.folder_ground_truth)
dict_notannotated_path = image_path_to_dict(args.folder_root + "/" + args.folder_not_annotated)
df = pd.read_csv (args.allrovcomb5_csv, sep =',')



class_dict = create_class_dict (dict_images_path, dict_groundtruth_path, df)

list_train, list_test, list_validate, list_all = class_dict_to_datasets (class_dict, float(args.train_percentage), float(args.test_percentage))


print("Train-Set: " + str(len(list_train)))
save_json(args.folder_root + "/" + args.train_json, list_train)

print("Test-Set: " + str(len(list_test)))
save_json(args.folder_root + "/" + args.test_json, list_test)

print("Validate-Set: " + str(len(list_validate)))
save_json(args.folder_root + "/" + args.validate_json, list_validate)

list_notannotated = create_notannotated_dict(dict_notannotated_path, df)
print("All-Set: " + str(len(list_all)) + " annotated " + str(len(list_notannotated)) + " not annotated")
save_json(args.folder_root + "/" + args.all_json, list_all + list_notannotated) # merge lists
