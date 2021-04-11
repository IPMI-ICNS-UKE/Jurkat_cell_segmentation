#!/usr/bin/env python
# coding: utf-8

""" Loads a reservoir computing model on streams of Ca Image data
measured with Fluo-4 (green) emission to segmen/classify existing
objects (T-cells and beads). """

__author__ = 'f.hadaeghi@uke.de'


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import grey_dilation, grey_erosion, binary_fill_holes
from scipy.ndimage import binary_fill_holes
from skimage import morphology

from echoes import ESNRegressor
from echoes.plotting import set_mystyle
from echoes.reservoir._leaky_numba import harvest_states
from numba import njit

from scipy.special import softmax

import pickle

# load the trained model
with open('trainedESN_Encoding_01.pickle', 'rb') as handle:
    esn = pickle.load(handle)
print(esn)


# define functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
                img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
                img = cv2.medianBlur(img, 5)
                img = (img - img.mean()) / (img.std())
                img = sigmoid(2 * img)
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
            img = sigmoid(2 * img)
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
    dilation = grey_erosion(pre_s.astype(np.int32), structure=er_strel.astype(np.int8))
    dilation = grey_dilation(dilation.astype(np.int32), structure=strel.astype(np.int8))
    cleanMask = np.float32(morphology.remove_small_objects(dilation.astype(np.bool),
                                                           min_size=200, connectivity=8))
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
            Y[i, :, :] = np.sum(X[i:, :, :, ], axis=0)/((DataLength-i))
    return Y


def frame_reconstruct(y_prdct, nRow, nCol):
    numFrame = int(y_prdct.shape[0] / (nRow*nCol))
    pr_output = np.reshape(y_prdct, (numFrame, nRow, nCol)).astype(np.float64)
    return pr_output


def prediction_resize(prdct, resize_factor):
    [nFrame, nRows, nCols] = prdct.shape
    prdct_resize = np.zeros((nFrame, nRows*resize_factor, nCols*resize_factor))
    for frame in range(nFrame):
        prdct_resize[frame, :, :] = cv2.resize(prdct[frame, :, :], None, fx=resize_factor,
                                               fy=resize_factor, interpolation=cv2.INTER_CUBIC)
    prdct = mymovmen(prdct_resize, 30)  # Optional (improves temporal consistency)
#     prdct = (prdct - prdct.mean()) / (prdct.std())
#     prdct = (prdct_resize - prdct_resize.mean()) / (prdct_resize.std())
    return prdct


def stack_post_process(prdct, strel, er_strel, kernel, thr):
    binary_prdct = np.zeros(prdct.shape)
    for frame in range(prdct.shape[0]):
        binary_prdct[frame, :, :] = PostProcess(prdct[frame, :, :], strel, er_strel, kernel, thr)
    return binary_prdct


def adjust_brightness(image, delta):
    out_img = image + delta
    return out_img


def adjust_contrast(image, factor):
    img_mean = image.mean()
    out_img = (image - img_mean) * factor + img_mean
    return out_img

root_dir = '.~/Task3/RC/Data'
base_dir = str(root_dir)

sequence_dirs = ['/Test/07/']
target_dirs = ['/Test_GT/07_GT/']
ImVect, MaskVect, nRow, nCol = input_output_generation(sequence_dirs, target_dirs)
AugmentedIm = np.hstack((ImVect, np.hstack((np.roll(ImVect, 1, axis=0), np.roll(ImVect, -1, axis=0)))))
inputs = AugmentedIm.astype(np.float64)
outputs = MaskVect.astype(np.float64)
y_pred = esn.predict(inputs)


kernel = np.ones((5, 5), np.float32) / 25
strel = np.zeros((4, 4))
er_strel = np.zeros((2, 2))

# aplly softmax
y_pred_smax = softmax(y_pred, axis=-1)

# reconstruct the frame
predict_class = np.argmax(y_pred_smax, axis=1)
predict_class = np.array(predict_class.tolist())

pr_cell = np.zeros(predict_class.shape)
pr_cell[predict_class == 0] = 1
pr_bead = np.zeros(predict_class.shape)
pr_bead[predict_class == 1] = 1
pr_background = np.zeros(predict_class.shape)
pr_background[predict_class == 2] = 1

pr_output_cell = frame_reconstruct(pr_cell, nRow, nCol)
pr_output_bead = frame_reconstruct(pr_bead, nRow, nCol)
pr_output_background = frame_reconstruct(pr_background, nRow, nCol)

prdct_resize_cell = prediction_resize(pr_output_cell, 5)
prdct_resize_bead = prediction_resize(pr_output_bead, 5)
prdct_resize_background = prediction_resize(pr_output_background, 5)
prdct_resize_binary_cell = stack_post_process(prdct_resize_cell, strel, er_strel, kernel, 0.3)
prdct_resize_binary_bead = stack_post_process(prdct_resize_bead, strel, er_strel, kernel, 0.3)
prdct_resize_binary_background = stack_post_process(prdct_resize_background, strel, er_strel, kernel, 0.3)

# save predictions
path = root_dir + str("/Predictions/")


try:
    os.mkdir(path)
    path_softmax = path + str("/Softmax_Results/")
    path_mask = path + str("/Mask_Results/")
    os.mkdir(path_softmax)
    os.mkdir(path_mask)
except OSError:
    print("Creation of the directory %s failed" % path)
else:
    print("Successfully created the directory %s " % path)

for frame in range(prdct_resize_cell.shape[0]):
    # Filename
    filename = str("t%03d.png" % frame)
    prdct_cell = prdct_resize_cell[frame, :, :]
    prdct_bead = prdct_resize_bead[frame, :, :]
    prdct_background = prdct_resize_background[frame, :, :]

    pair_img = 255 * np.dstack((prdct_cell, prdct_bead, prdct_background))
    img = 255 * prdct_resize_binary_cell[frame, :, :]
    img[prdct_resize_binary_bead[frame, :, :] > 0] = 128
    # Saving the image
    cv2.imwrite(str(path_softmax)+str(filename), pair_img)
    cv2.imwrite(str(path_mask)+str(filename), img)
