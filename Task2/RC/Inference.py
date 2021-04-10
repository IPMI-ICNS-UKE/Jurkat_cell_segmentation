#!/usr/bin/env python
# coding: utf-8

""" Loads trained reservoir computing models and streams of
Ca Image data measured with Flu-4 (green) and Fura Red (red)
emissions and segmen/classify existing objects
(T-cells and beads). """

__author__ = 'f.hadaeghi@uke.de'

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import grey_dilation, grey_erosion
from scipy.ndimage import binary_fill_holes
from skimage import morphology
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split

from echoes import ESNRegressor
from echoes.plotting import set_mystyle
from echoes.reservoir._leaky_numba import harvest_states
from numba import njit

from pystackreg import StackReg

import pickle

# load RC model (Green)
with open('trainedESN_Green.pickle', 'rb') as handle:
    esn_green = pickle.load(handle)

# load RC model (Red)
with open('trainedESN_Red.pickle', 'rb') as handle:
    esn_red = pickle.load(handle)

print(esn_green)
print(esn_red)


# functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def PystackMethods_Reg(ref, mov, mode):
    if mode == "Translational":
        sr = StackReg(StackReg.TRANSLATION)
        corrected_image = sr.register_transform(ref, mov)

    if mode == "Rigid":
        sr = StackReg(StackReg.RIGID_BODY)
        corrected_image = sr.register_transform(ref, mov)

    if mode == "Scaled Rotation":
        sr = StackReg(StackReg.SCALED_ROTATION)
        corrected_image = sr.register_transform(ref, mov)

    if mode == "Affine":
        sr = StackReg(StackReg.AFFINE)
        corrected_image = sr.register_transform(ref, mov)

    if mode == "Bilinear":
        sr = StackReg(StackReg.BILINEAR)
        corrected_image = sr.register_transform(ref, mov)

    return corrected_image


def Input_Output_Generation(sequence_dirs, target_dirs):
    for seq in range(len(sequence_dirs)):
        sequence_dir_green = str(base_dir_green) + str(sequence_dirs[seq])
        sequence_dir_red = str(base_dir_red) + str(sequence_dirs[seq])
        target_dir = str(base_dir_green) + str(target_dirs[seq])
        fnames = os.listdir(sequence_dir_green)
        for frame in range(0, len(fnames)):
            fileName = str("t%03d.tif" % frame)
            mskName = str("man_seg%03d.tif" % frame)
            if (seq == frame == 0):
                img = np.array(cv2.imread(str(sequence_dir_green + fileName), 0))
                mov = np.array(cv2.imread(str(sequence_dir_red + fileName), 0))
                # Registration
                corrected_img = PystackMethods_Reg(img, mov, "Rigid")

                # green vector
                img = img.astype(np.float32)
                img = cv2.resize(img, None, fx=0.2, fy=0.2,
                                 interpolation=cv2.INTER_CUBIC)
                img = cv2.medianBlur(img, 5)
                img = (img - img.mean()) / (img.std())
                img = sigmoid(img)
                ColIm_green = np.reshape(img, (-1, 1))
                RowIm_green = np.reshape(np.transpose(img), (-1, 1))
                ImgVector_green = np.hstack((ColIm_green, RowIm_green))

                # corrected red vector
                corrected_img = corrected_img.astype(np.float32)
                corrected_img = cv2.resize(corrected_img, None, fx=0.2, fy=0.2,
                                           interpolation=cv2.INTER_CUBIC)
                corrected_img = cv2.medianBlur(corrected_img, 5)
                corrected_img = (corrected_img - corrected_img.mean()) / (corrected_img.std())
                corrected_img = sigmoid(corrected_img)
                ColIm_red = np.reshape(corrected_img, (-1, 1))
                RowIm_red = np.reshape(np.transpose(corrected_img), (-1, 1))
                ImgVector_red = np.hstack((ColIm_red, RowIm_red))
                nRow, nCol = corrected_img.shape

                tmp_msk = np.array(cv2.imread(str(target_dir + mskName), 0))
                tmp_msk = cv2.resize(tmp_msk, None, fx=0.2, fy=0.2,
                                     interpolation=cv2.INTER_CUBIC)
                tmp_msk[tmp_msk < 1.5] = 0
                msk = np.reshape(tmp_msk, (-1, 1))

            img = np.array(cv2.imread(str(sequence_dir_green + fileName), 0))
            mov = np.array(cv2.imread(str(sequence_dir_red + fileName), 0))
            # Registration
            corrected_img = PystackMethods_Reg(img, mov, "Rigid")

            # green vector
            img = mg.astype(np.float32)
            img = cv2.resize(img, None, fx=0.2, fy=0.2,
                             interpolation=cv2.INTER_CUBIC)
            img = cv2.medianBlur(img, 5)
            img = (img - img.mean()) / (img.std())
            img = sigmoid(img)
            ColIm_green = np.reshape(img, (-1, 1))
            RowIm_green = np.reshape(np.transpose(img), (-1, 1))
            tmpVector_green = np.hstack((ColIm_green, RowIm_green))

            corrected_img = corrected_img.astype(np.float32)
            corrected_img = cv2.resize(corrected_img, None, fx=0.2, fy=0.2,
                                       interpolation=cv2.INTER_CUBIC)
            corrected_img = cv2.medianBlur(corrected_img, 5)
            corrected_img = (corrected_img - corrected_img.mean()) / (corrected_img.std())
            corrected_img = sigmoid(corrected_img)
            ColIm_red = np.reshape(corrected_img, (-1, 1))
            RowIm_red = np.reshape(np.transpose(corrected_img), (-1, 1))
            tmpVector_red = np.hstack((ColIm_red, RowIm_red))

            ImgVector_green = np.vstack((ImgVector_green, tmpVector_green))
            ImgVector_red = np.vstack((ImgVector_red, tmpVector_red))

            tmp_msk = np.array(cv2.imread(str(target_dir + mskName), 0))
            tmp_msk = cv2.resize(tmp_msk, None, fx=0.2, fy=0.2,
                                 interpolation=cv2.INTER_CUBIC)
            tmp_msk[tmp_msk < 1.5] = 0
            tmp_msk_vec = np.reshape(tmp_msk, (-1, 1))
            msk = np.vstack((msk, tmp_msk_vec))
        print('sequence', sequence_dirs[seq], 'completed')
    return ImgVector_green, ImgVector_red, msk, nRow, nCol


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def PostProcess(img, strel, er_strel, kernel, thr):
    pre_s = np.zeros(img.shape)
    pre_s[img >= thr] = 1
#     dilation = grey_erosion(pre_s.astype(np.int32), structure=er_strel.astype(np.int8))
#     dilation = grey_dilation(dilation.astype(np.int32), structure=strel.astype(np.int8))
    cleanMask = np.float32(morphology.remove_small_objects(pre_s.astype(np.bool),
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

# set the paths
root_dir_green = '.~/Task2/RC/Data/Green'  # green emission
root_dir_red = '.~/Task2/RC/Data/Red'  # red emission


base_dir_green = str(root_dir_green + '/Test/')
base_dir_red = str(root_dir_red + '/Test/')
sequence_dir_green = str(base_dir_green) + '07/'
sequence_dir_red = str(base_dir_red) + '07/'
target_dir = str(root_dir_green) + '/Test_GT/07_GT/'

fnames = os.listdir(sequence_dir_green)
for frame in range(0, len(fnames)):
    fileName = str("t%03d.tif" % frame)
    mskName = str("man_seg%03d.tif" % frame)
    if (frame == 0):
        img = np.array(cv2.imread(str(sequence_dir_green + fileName), 0))
        mov = np.array(cv2.imread(str(sequence_dir_red + fileName), 0))
        # Registration
        corrected_img = PystackMethods_Reg(img, mov, "Rigid")

        # green vector
        img = img.astype(np.float32)
        img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
        img = cv2.medianBlur(img, 5)
        img = (img - img.mean()) / (img.std())
        img = sigmoid(img)
        ColIm_green = np.reshape(img, (-1, 1))
        RowIm_green = np.reshape(np.transpose(img), (-1, 1))
        ImgVector_green = np.hstack((ColIm_green, RowIm_green))

        # corrected red vector
        corrected_img = corrected_img.astype(np.float32)
        corrected_img = cv2.resize(corrected_img, None, fx=0.2, fy=0.2,
                                   interpolation=cv2.INTER_CUBIC)
        corrected_img = cv2.medianBlur(corrected_img, 5)
        corrected_img = (corrected_img - corrected_img.mean()) / (corrected_img.std())
        corrected_img = sigmoid(corrected_img)
        ColIm_red = np.reshape(corrected_img, (-1, 1))
        RowIm_red = np.reshape(np.transpose(corrected_img), (-1, 1))
        ImgVector_red = np.hstack((ColIm_red, RowIm_red))
        nRow, nCol = corrected_img.shape

        tmp_msk = np.array(cv2.imread(str(target_dir + mskName), 0))
        tmp_msk = cv2.resize(tmp_msk, None, fx=0.2, fy=0.2,
                             interpolation=cv2.INTER_CUBIC)
        tmp_msk[tmp_msk < 1.5] = 0
        msk = np.reshape(tmp_msk, (-1, 1))

    img = np.array(cv2.imread(str(sequence_dir_green + fileName), 0))
    mov = np.array(cv2.imread(str(sequence_dir_red + fileName), 0))
    # Registration
    corrected_img = PystackMethods_Reg(img, mov, "Rigid")

    # green vector
    img = img.astype(np.float32)
    img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.medianBlur(img, 5)
    img = (img - img.mean()) / (img.std())
    img = sigmoid(img)
    ColIm_green = np.reshape(img, (-1, 1))
    RowIm_green = np.reshape(np.transpose(img), (-1, 1))
    tmpVector_green = np.hstack((ColIm_green, RowIm_green))

    corrected_img = corrected_img.astype(np.float32)
    corrected_img = cv2.resize(corrected_img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    corrected_img = cv2.medianBlur(corrected_img, 5)
    corrected_img = (corrected_img - corrected_img.mean()) / (corrected_img.std())
    corrected_img = sigmoid(corrected_img)
    ColIm_red = np.reshape(corrected_img, (-1, 1))
    RowIm_red = np.reshape(np.transpose(corrected_img), (-1, 1))
    tmpVector_red = np.hstack((ColIm_red, RowIm_red))

    ImgVector_green = np.vstack((ImgVector_green, tmpVector_green))
    ImgVector_red = np.vstack((ImgVector_red, tmpVector_red))

    tmp_msk = np.array(cv2.imread(str(target_dir + mskName), 0))
    tmp_msk = cv2.resize(tmp_msk, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    tmp_msk[tmp_msk < 1.5] = 0

    tmp_msk_vec = np.reshape(tmp_msk, (-1, 1))
    msk = np.vstack((msk, tmp_msk_vec))

AugmentedIm_green = np.hstack((ImgVector_green, np.hstack((np.roll(ImgVector_green, 1, axis=0),
                                                          np.roll(ImgVector_green, -1, axis=0)))))
AugmentedIm_red = np.hstack((ImgVector_red, np.hstack((np.roll(ImgVector_red, 1, axis=0),
                                                      np.roll(ImgVector_red, -1, axis=0)))))
inputs_green = AugmentedIm_green.astype(np.float64)
inputs_red = AugmentedIm_red.astype(np.float64)
outputs = msk.astype(np.float64)

X_test_green = inputs_green
X_test_red = inputs_red
y_test = outputs/2

# one option to improve segmention results is to repeat the input time-series
# and then throw away the first half of predicted time-series, for instance:
# X_test = np.tile(inputs, (2,1))
# y_test = np.tile(outputs, (2,1))

y_pred_green = esn_green.predict(X_test_green)
y_pred_red = esn_red.predict(X_test_red)

# In case the time-series is repeated once
# index = np.arange(int(y_pred.shape[0]/2)).reshape(int(y_pred.shape[0]/2),1)
# y_pred = np.delete(y_pred, index)
# y_test = np.delete(y_test, index)
# print (y_pred.shape[0])

kernel = np.ones((3, 3), np.float32) / 9
strel = np.zeros((3, 3))
er_strel = np.zeros((2, 2))

# reconstruct and resize frames
tg_output, pr_output_green = frame_reconstruct(y_test, y_pred_green, nRow, nCol)
tg_resize, prdct_resize_green = prediction_resize(tg_output, pr_output_green, 5)
prdct_resize_binary_green = stack_post_process(prdct_resize_green, strel, er_strel, kernel, 0.65)
# red
tg_output, pr_output_red = frame_reconstruct(y_test, y_pred_red, nRow, nCol)
tg_resize, prdct_resize_red = prediction_resize(tg_output, pr_output_red, 5)
prdct_resize_binary_red = stack_post_process(prdct_resize_red, strel, er_strel, kernel, 0.5)

Cell_pred = prdct_resize_binary_red
Bead_pred = np.logical_xor(prdct_resize_binary_green, prdct_resize_binary_red)

Final_pred = np.zeros(prdct_resize_binary_green.shape)
Final_pred[prdct_resize_binary_red > 0] = 1
Final_pred[Bead_pred > 0] = 0.5

# save predictions
root_dir = '.~/Task2/RC'
path = root_dir + str("/Predictions/")


try:
    os.mkdir(path)

except OSError:
    print("Creation of the directory %s failed" % path)
else:
    print("Successfully created the directory %s " % path)

for frame in range(Final_pred.shape[0]):
    # Filename
    filename = str("t%03d.png" % frame)
    # Saving the image
    cv2.imwrite(str(path)+str(filename), 255*Final_pred[frame, :, :])
