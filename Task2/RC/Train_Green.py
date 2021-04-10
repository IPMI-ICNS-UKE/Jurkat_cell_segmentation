#!/usr/bin/env python
# coding: utf-8

""" Trains a reservoir computing model on streams of Ca Image data
measured with Fluo-4 (green) emission to segmen existing objects
(T-cells and beads). """

__author__ = 'f.hadaeghi@uke.de'

# imports
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import grey_dilation, affine_transform, grey_erosion
from scipy.ndimage import binary_fill_holes
from skimage import morphology
# for coross-validation
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split

from echoes import ESNRegressor
from echoes.plotting import set_mystyle
from echoes.reservoir._leaky_numba import harvest_states
from numba import njit

import pickle

# set paths
root_dir = '.~/Task2/RC/Data/Green'  # green emission

base_dir = str(root_dir + '/Training/')
sequence_dir = str(base_dir) + '01/'
target_dir = str(base_dir) + 'GT_01/'
print(sequence_dir)


# functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Input_Output_Generation(sequence_dirs, target_dirs):
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
                img = sigmoid(img)
                ColIm = np.reshape(img, (-1, 1))
                RowIm = np.reshape(np.transpose(img), (-1, 1))
                ImgVector = np.hstack((ColIm, RowIm))
                nRow, nCol = img.shape

                tmp_msk = np.array(cv2.imread(str(target_dir + mskName), 0))
                tmp_msk = cv2.resize(tmp_msk, None, fx=0.2, fy=0.2,
                                     interpolation=cv2.INTER_CUBIC)
                tmp_msk[tmp_msk >= 0.5] = 1
                tmp_msk[tmp_msk < 0.5] = 0
                msk = np.reshape(tmp_msk, (-1, 1))

            img = np.array(cv2.imread(str(sequence_dir + fileName), 0))
            img = img.astype(np.float32)
            img = cv2.resize(img, None, fx=0.2, fy=0.2,
                             interpolation=cv2.INTER_CUBIC)
            img = cv2.medianBlur(img, 5)
            img = (img - img.mean()) / (img.std())
            img = sigmoid(img)

            ColIm = np.reshape(img, (-1, 1))
            RowIm = np.reshape(np.transpose(img), (-1, 1))
            tmpVector = np.hstack((ColIm, RowIm))
            ImgVector = np.vstack((ImgVector, tmpVector))
            tmp_msk = np.array(cv2.imread(str(target_dir + mskName), 0))
            tmp_msk = cv2.resize(tmp_msk, None, fx=0.2, fy=0.2,
                                 interpolation=cv2.INTER_CUBIC)
            tmp_msk[tmp_msk >= 0.5] = 1
            tmp_msk[tmp_msk < 0.5] = 0

            tmp_msk_vec = np.reshape(tmp_msk, (-1, 1))
            msk = np.vstack((msk, tmp_msk_vec))
        print('sequence', sequence_dirs[seq], 'completed')
    return ImgVector, msk, nRow, nCol


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def PostProcess(img, strel, er_strel, kernel, thr):
    pre_s = np.zeros(img.shape)
    pre_s[img >= thr] = 1
#     dilation = grey_erosion(pre_s.astype(np.int32), structure=er_strel.astype(np.int8))
#     dilation = grey_dilation(dilation.astype(np.int32), structure=strel.astype(np.int8))
    cleanMask = np.float32(morphology.remove_small_objects(pre_s.astype(np.bool),
                                                           min_size=100, connectivity=8))
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
    ValidLength = int(y_trgt.shape[0] / 1)
    gt = y_trgt[-ValidLength-1:-1]
    pred = y_prdct[-ValidLength-1:-1]
    numFrame = int(gt.shape[0] / (nRow*nCol))
    start_pixel = np.mod(gt.shape[0], nRow*nCol)
    tg_output = np.reshape(gt[-numFrame*nRow*nCol-1:-1], (numFrame, nRow, nCol)).astype(np.float64)
    pr_output = np.reshape(pred[-numFrame*nRow*nCol-1:-1], (numFrame, nRow, nCol)).astype(np.float64)
    return tg_output, pr_output


def prediction_resize(tg, prdct, resize_factor):
    [nFrame, nRows, nCols] = tg.shape
    tg_resize = np.zeros((nFrame, nRows*resize_factor, nCols*resize_factor))
    prdct_resize = np.zeros((nFrame, nRows*resize_factor, nCols*resize_factor))
    for frame in range(nFrame):
        tg_resize[frame, :, :] = cv2.resize(tg[frame, :, :], None, fx=resize_factor,
                                            fy=resize_factor, interpolation=cv2.INTER_CUBIC)
        prdct_resize[frame, :, :] = cv2.resize(prdct[frame, :, :], None, fx=resize_factor,
                                               fy=resize_factor, interpolation=cv2.INTER_CUBIC)
    prdct = mymovmen(prdct_resize, 5)  # Optional (improves temporal consistency)
    # optional normalization
    prdct = (prdct - prdct.min()) / (prdct.max()-prdct.min())
#     prdct = (prdct_resize - prdct_resize.mean()) / (prdct_resize.std())

    tg_resize[tg_resize >= 0.5] = 1
    tg_resize[tg_resize < 0.5] = 0
    return tg_resize, prdct


def stack_post_process(prdct, strel, er_strel, kernel, thr):
    binary_prdct = np.zeros(prdct.shape)
    for frame in range(prdct.shape[0]):
        binary_prdct[frame, :, :] = PostProcess(prdct[frame, :, :], strel, er_strel, kernel, thr)
    return binary_prdct

# read the training sequences and generate input-output vectors
sequence_dirs = ['01/', '02/', '03/', '04/', '05/']
target_dirs = ['01_GT/SEG/', '02_GT/SEG/', '03_GT/SEG/', '04_GT/SEG/', '05_GT/SEG/']
ImVector, msk, nRow, nCol = Input_Output_Generation(sequence_dirs, target_dirs)
AugmentedIm = np.hstack((ImVector, np.hstack((np.roll(ImVector, 1, axis=0), np.roll(ImVector, -1, axis=0)))))
gt_vect = msk


# Design the RC system and split the data
@njit
def mysigmoid(x):
    return (1.0/(1.0 + np.exp(-x)))


@njit
def myrelu(x):
    return (np.maximum(x, 0))


set_mystyle()  # optional: set aesthetics

inputs = AugmentedIm.astype(np.float64)
weight = 10
outputs = gt_vect.astype(np.float64)
X_train = inputs
X_test = inputs
y_train = weight * outputs
y_test = outputs

esn = ESNRegressor(
    n_reservoir=100,
    spectral_radius=.99,
    leak_rate=1.0,
    noise=0.0001,
    n_transient=0,
    input_scaling=1.0,
    input_shift=0.0,
    sparsity=0.98,
    bias=0.2,
    # activation_out = mysigmoid,
    # activation=myrelu,
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


kernel = np.ones((3, 3), np.float32) / 9
strel = np.zeros((3, 3))
er_strel = np.zeros((2, 2))
thr = 0.65  # Green

tg_output, pr_output = frame_reconstruct(y_test, y_pred, nRow, nCol)
tg_resize, prdct_resize = prediction_resize(tg_output, pr_output, 5)
prdct_resize_binary = stack_post_process(prdct_resize, strel, er_strel, kernel, thr)

# save the model
with open('trainedESN_Green.pickle', 'wb') as handle:
    pickle.dump(esn, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('trainedESN_Green.pickle', 'rb') as handle:
#     b = pickle.load(handle)

# print (b)
