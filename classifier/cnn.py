import cv2
import numpy as np
import tensorflow as tf


SINGLE = 0
MULTIPLE = 1

def ini (folder):
    '''Initialisierung

    Args:


    Returns:

    '''
    with tf.gfile.GFile(folder, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def features(X, mode):
    '''Feature(s) extrahieren

    Args:
        X (list):  Patch or Patches
        mode: SINGLE, MULTIPLE

    Returns:
        features (list)
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        # nur einmal ausführen
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0') # ---------------------------- aus dem pool_3:0 Layer werden die Features herausgeholt

        if (mode == MULTIPLE): # Mehrere Features
            listFeatures = []
            for Patch in X:
                feat = sess.run(next_to_last_tensor,{'DecodeJpeg:0': Patch})
                feat = np.squeeze(np.array(feat)) # In array überführen, # Eine Dimension zuviel, diese entfernen
                listFeatures.append(feat)
            return listFeatures

        if (mode == SINGLE): # Nur ein Feature
            feat = sess.run(next_to_last_tensor,{'DecodeJpeg:0': X})
            feat = np.squeeze(np.array(feat)) # In array überführen, # Eine Dimension zuviel, diese entfernen
            return feat


def featurelist(path_train_set):
    '''Trainings Patches laden und Features extrahieren

    Args:
        path_train_set (list): Pfade auf Patches


    Returns:
        features (list)

    '''
    listFeatures = []

    for index, Path in enumerate(path_train_set):
        loaded_patch = cv2.imread(Path)
        listFeatures.append(features(loaded_patch, SINGLE))

    return listFeatures