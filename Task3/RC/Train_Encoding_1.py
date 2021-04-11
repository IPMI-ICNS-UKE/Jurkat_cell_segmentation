#!/usr/bin/env python
# coding: utf-8

""" Trains a reservoir computing model on streams of Ca Image data
measured with Fluo-4 (green) emission to segmen/classify
existing objects (T-cells and beads). """

__author__ = 'f.hadaeghi@uke.de'


# imports
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import grey_dilation, affine_transform
from scipy.ndimage import grey_erosion, binary_fill_holes
from skimage import morphology
# for cross-validation
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split

from echoes import ESNRegressor
from echoes.plotting import set_mystyle
from echoes.reservoir._leaky_numba import harvest_states
from numba import njit

from scipy.special import softmax

import pickle

# set paths
root_dir = '.~/Task3/RC/Data'
base_dir = str(root_dir + '/Training/')
sequence_dir = str(base_dir) + '01/'
target_dir = str(base_dir) + '01_GT/'
print(sequence_dir)


# define the functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def adjust_brightness(image, delta):
    out_img = image + delta
    return out_img


def adjust_contrast(image, factor):
    img_mean = image.mean()
    out_img = (image - img_mean) * factor + img_mean
    return out_img


def input_output_generation(sequence_dirs, target_dirs):
    for seq in range(len(sequence_dirs)):
        sequence_dir = str(base_dir) + str(sequence_dirs[seq])
        target_dir = str(base_dir) + str(target_dirs[seq])
        fnames = os.listdir(sequence_dir)
        for frame in range(0, len(fnames)):
            fileName = str("t%03d.tif" % frame)
            mskName = str("man_seg%03d.tif" % frame)
            if (seq == frame == 0):
                img = np.array(cv2.imread(str(sequence_dir + fileName), 0))
                img = img.astype(np.float32)
                img = cv2.resize(img, None, fx=0.2, fy=0.2,
                                 interpolation=cv2.INTER_CUBIC)
                img = cv2.medianBlur(img, 5)
                img = (img - img.mean()) / (img.std())
                img = sigmoid(2*img)
#                 random_constrast_factor = np.random.rand() + 0.5
#                 random_brightness_delta = (np.random.rand() - 0.5) * 0.2 * img.max()
#                 img = adjust_brightness(img, random_brightness_delta)
#                 img = adjust_contrast(img, random_constrast_factor)
                ColIm = np.reshape(img, (-1, 1))
                RowIm = np.reshape(np.transpose(img), (-1, 1))
                ImgVector = np.hstack((ColIm, RowIm))
                nRow, nCol = img.shape

                tmp_msk = np.array(cv2.imread(str(target_dir + mskName), 0))
                tmp_msk = cv2.resize(tmp_msk, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)

                msk = np.reshape(tmp_msk, (-1, 1))

            img = np.array(cv2.imread(str(sequence_dir + fileName), 0))
            img = img.astype(np.float32)
            img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
            img = cv2.medianBlur(img, 5)
            img = (img - img.mean()) / (img.std())
            img = sigmoid(2*img)
#             random_constrast_factor = np.random.rand() + 0.5
#             random_brightness_delta = (np.random.rand() - 0.5) * 0.2 * img.max()
#             img = adjust_brightness(img, random_brightness_delta)
#             img = adjust_contrast(img, random_constrast_factor)

            ColIm = np.reshape(img, (-1, 1))
            RowIm = np.reshape(np.transpose(img), (-1, 1))
            tmpVector = np.hstack((ColIm, RowIm))
            ImgVector = np.vstack((ImgVector, tmpVector))
            tmp_msk = np.array(cv2.imread(str(target_dir + mskName), 0))
            tmp_msk = cv2.resize(tmp_msk, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)

            tmp_msk_vec = np.reshape(tmp_msk, (-1, 1))
            msk = np.vstack((msk, tmp_msk_vec))
        print('sequence', sequence_dirs[seq], 'completed')
    return ImgVector, msk, nRow, nCol


def PostProcess(img, strel, er_strel, kernel, thr):
    pre_s = np.zeros(img.shape)
    pre_s[img >= thr] = 1
#     dilation = grey_erosion(pre_s.astype(np.int32), structure=er_strel.astype(np.int8))
#     dilation = grey_dilation(dilation.astype(np.int32), structure=strel.astype(np.int8))
    cleanMask = np.float32(morphology.remove_small_objects(pre_s.astype(np.bool),
                                                           min_size=50, connectivity=8))
    cleanMask = binary_fill_holes(cleanMask)
    cleanMask = cv2.filter2D(cleanMask.astype(np.float32), -1, kernel)
    cleanMask[cleanMask >= 0.5] = 1
    cleanMask[cleanMask < 0.5] = 0
    return cleanMask


def mymovmen(X, n):
    DataLength = X.shape[0]
    Y = np.zeros(X.shape)
    for i in range(DataLength):
        if i <= (DataLength-n):
            Y[i, :, :] = np.sum(X[i:i+n+1, :, :, ], axis=0) / n
        else:
            Y[i, :, :] = np.sum(X[i:, :, :, ], axis=0) / ((DataLength-i))
    return Y


def frame_reconstruct(y_trgt, y_prdct, nRow, nCol):
    gt = y_trgt
    pred = y_prdct
    numFrame = int(gt.shape[0] / (nRow*nCol))
    tg_output = np.reshape(gt, (numFrame, nRow, nCol)).astype(np.float64)
    pr_output = np.reshape(pred, (numFrame, nRow, nCol)).astype(np.float64)
    return tg_output, pr_output


def prediction_resize(tg, prdct, resize_factor):
    [nFrame, nRows, nCols] = tg.shape
    tg_resize = np.zeros((nFrame, nRows*resize_factor, nCols*resize_factor))
    prdct_resize = np.zeros((nFrame, nRows*resize_factor, nCols*resize_factor))
    tg_resize[frame, :, :] = cv2.resize(tg[frame, :, :], None, fx=resize_factor,
                                        fy=resize_factor, interpolation=cv2.INTER_CUBIC)
    prdct_resize[frame, :, :] = cv2.resize(prdct[frame, :, :], None, fx=resize_factor,
                                           fy=resize_factor, interpolation=cv2.INTER_CUBIC)
    prdct = mymovmen(prdct_resize, 10)  # Optional (improves temporal consistency)
    prdct = (prdct - prdct.mean()) / (prdct.std())
#     prdct = (prdct_resize - prdct_resize.mean()) / (prdct_resize.std())

    tg_resize[tg_resize >= 0.5] = 1
    tg_resize[tg_resize < 0.5] = 0
    return tg_resize, prdct_resize


def stack_post_process(prdct, strel, er_strel, kernel, thr):
    binary_prdct = np.zeros(prdct.shape)
    for frame in range(prdct.shape[0]):
        binary_prdct[frame, :, :] = PostProcess(prdct[frame, :, :], strel, er_strel, kernel, thr)
    return binary_prdct

# read the training sequences and generate input-output vectors
sequence_dirs = ['01/', '02/', '03/', '04/', '05/']
target_dirs = ['01_GT/', '02_GT/', '03_GT/', '04_GT/', '05_GT/']
ImVect, MaskVect, nRow, nCol = input_output_generation(sequence_dirs, target_dirs)
AugmentedIm = np.hstack((ImVect, np.hstack((np.roll(ImVect, 1, axis=0), np.roll(ImVect, -1, axis=0)))))

# create Output
out_1 = np.zeros(MaskVect.shape)
out_2 = np.zeros(MaskVect.shape)
out_3 = np.zeros(MaskVect.shape)

out_1[MaskVect == 255] = 1  # cell
out_2[MaskVect == 128] = 1  # bead
out_3[MaskVect == 0] = 1  # background

# class 0: cell, class 1: bead, class 2: background
OutputVect = np.hstack((out_1, np.hstack((out_2, out_3))))

# Design the RC system and split the data


@njit
def mysigmoid(x):
    return (1.0/(1.0 + np.exp(-x)))


@njit
def myrelu(x):
    return (np.maximum(x, 0))


@njit
def mytanh(x):
    return np.tanh(x)

set_mystyle()  # optional: set aesthetics


X_train = AugmentedIm
X_test = AugmentedIm
y_train = OutputVect
y_test = OutputVect

# X_train = np.tile(X_train, (2,1))
# y_train = np.tile(y_train, (2,1))
# X_test = np.tile(X_test, (2,1))
# y_test = np.tile(y_test, (2,1))

class_weights = [10, 15, 8]
y_train = class_weights * y_train
esn = ESNRegressor(
    n_reservoir=500,
    spectral_radius=0.99,
    leak_rate=.4,
    noise=0.0001,
    n_transient=0,
    input_scaling=1.0,
    input_shift=0.0,
    sparsity=0.98,
    bias=0.0,
    # activation_out = mysigmoid,
    # activation=mytanh,
    # n_transient=500*250,
    regression_method="ridge",
    ridge_alpha=0.000001,
    random_state=42,
    ridge_normalize=False,
    ridge_fit_intercept=False,
    store_states_train=False,
    store_states_pred=False,
)

esn.fit(X_train, y_train)
y_pred = esn.predict(X_test)


thr = 0.5
kernel = np.ones((5, 5), np.float32) / 25
strel = np.zeros((4, 4))
er_strel = np.zeros((2, 2))

# aplly softmax
y_pred_smax = softmax(y_pred, axis=-1)

# reconstruct the frame
tg_cell = y_test[:, 0]
tg_bead = y_test[:, 1]

predict_class = np.argmax(y_pred_smax, axis=1)
predict_class = np.array(predict_class.tolist())

pr_cell = np.zeros(predict_class.shape)
pr_cell[predict_class == 0] = 1
pr_bead = np.zeros(predict_class.shape)
pr_bead[predict_class == 1] = 1

tg_output_cell, pr_output_cell = frame_reconstruct(tg_cell, pr_cell, nRow, nCol)
tg_output_bead, pr_output_bead = frame_reconstruct(tg_bead, pr_bead, nRow, nCol)
tg_resize_cell, prdct_resize_cell = prediction_resize(tg_output_cell, pr_output_cell, 5)
tg_resize_bead, prdct_resize_bead = prediction_resize(tg_output_bead, pr_output_bead, 5)
prdct_resize_binary_cell = stack_post_process(prdct_resize_cell, strel, er_strel, kernel, 0.3)
prdct_resize_binary_bead = stack_post_process(prdct_resize_bead, strel, er_strel, kernel, 0.3)

# save the model
with open('trainedESN_Encoding_01.pickle', 'wb') as handle:
    pickle.dump(esn, handle, protocol=pickle.HIGHEST_PROTOCOL)
