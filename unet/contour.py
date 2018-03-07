import os
import sys
import random
import cv2
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

#矩阵加法
def madd(M1, M2):
    if isinstance(M1, (tuple, list)) and isinstance(M2, (tuple, list)):
        return [[m+n for m,n in zip(i,j)] for i, j in zip(M1,M2)]


# Set some parameters
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3
BATCH_SIZE = 16
TRAIN_PATH = '../../stage1_train/'
TEST_PATH = '../../stage1_test/'
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 3

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
train_ids = train_ids[:5]
test_ids = test_ids[:5]
np.random.seed(10)

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)

print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

im_tr = X_train[3]
im_te = Y_train[3]
im_tr = cv2.cvtColor(im_tr, cv2.COLOR_BGR2GRAY)
retval, im_tr1 = cv2.threshold(im_tr, 255*0.1, 255, cv2.THRESH_BINARY)
im_tr1 = np.maximum(im_tr, im_tr1)
# im_te = cv2.cvtColor(im_te, cv2.COLOR_BGR2GRAY)
# cv2.imshow('1',im_tr)
# cv2.imshow('2',im_te)
# cv2.waitKey(0)
image1, contours1, hierarchy1 = cv2.findContours(im_tr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # 检索模式为树形cv2.RETR_TREE，
#轮廓存储模式为简单模式cv2.CHAIN_APPROX_SIMPLE，如果设置为 cv2.CHAIN_APPROX_NONE，所有的边界点都会被存储。
res1 = cv2.drawContours(im_tr1, contours1, -1, 122, 1)
image2, contours2, hierarchy2 = cv2.findContours(im_te,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # 检索模式为树形cv2.RETR_TREE，
#轮廓存储模式为简单模式cv2.CHAIN_APPROX_SIMPLE，如果设置为 cv2.CHAIN_APPROX_NONE，所有的边界点都会被存储。
res2 = cv2.drawContours(im_te, contours2, -1, 122, 1)

cv2.imshow('0',im_tr1)
cv2.imshow('00',im_tr)
cv2.imshow('1',res1)
cv2.imshow('2',res2)
cv2.waitKey(0)