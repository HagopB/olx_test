import json
from scipy.misc import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
from glob import glob
import os

import keras
from keras.applications import VGG16, InceptionResNetV2, ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, Model

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


########################################################
####################### MODELS #########################
########################################################

# IQA model
def IQA(weights_path):
    """ getting the NIMA IQA pre-trained model """
    base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)

    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights(weights_path)
    return model

########################################################
#################### MODEL UTILS #######################
########################################################

# passing from imagenet snysets to custom binary case
def synsets2clocks(probas):
    """ Allows to pass from the 1000 synset probas to custom case
        Simple summing of labels belonging to watch/clock children synsets
    """
    clock_ids = [409, 530, 531, 826, 892]

    probas_clocks_ = []
    for idx, el in enumerate(probas):
        probas_clocks_.append(probas[idx, clock_ids].sum())

    return probas_clocks_

# compute the mean of image quality scores distribution
def mean_score(scores):
    """ computes the IQA mean score """
    si = np.arange(1, 11, 1)
    mean_ = np.sum(scores * si)
    return mean_

# compute the std of image quality scores distribution
def std_score(scores):
    """ computes the IQA std score
        Necessary for the AVA data set, not a must for us
    """
    si = np.arange(1, 11, 1)
    mean = mean_score(scores)
    std_ = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std_

########################################################
#################### LOAD IMAGES #######################
########################################################

# keras method for data generating
def get_batches(dirname,
                gen=image.ImageDataGenerator(),
                shuffle=False,
                batch_size=1,
                class_mode='categorical',
                target_size=(224,224)):
    """ Batches of images
        Batch data generator, ready to be fit or predict on (generator).
    """
    return gen.flow_from_directory(dirname, target_size=target_size,
                                   class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)
# load images from a given folder
def load_images(path_to_images):
    """
    It will load locally the images and transform them to arrays.
    It will output the convolutional features.
    """
    types = ['*.jpg','*.png','*.jpeg']

    listdir = []
    path = path_to_images
    for files in types:
        listdir.extend(glob(os.path.join(path, files)))

    # import images
    tmp = dict()
    for f in listdir:
        _id = f
        img = imread(f)
        img = imresize(img,(224,224,3))
        tmp[_id] = dict(image=img)

    X_test = np.array([tmp.get(k)['image'] for k in listdir])

    return X_test, listdir

if __name__ == "__main__":
    pass
