#!/usr/bin/env python
# coding: utf-8

""" Loads a reservoir computing model on streams of Ca Image data
measured with Fluo-4 (green) emission to segmen/classify existing
objects (T-cells and beads). """

__author__ = 'f.hadaeghi@uke.de'

# imports
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import grey_dilation, grey_erosion
from scipy.ndimage import binary_fill_holes
from skimage import morphology
from echoes import ESNRegressor
from echoes.plotting import set_mystyle
from echoes.reservoir._leaky_numba import harvest_states
from numba import njit

from scipy.special import softmax
import pickle


# define functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def adjust_brightness(image, delta):
    out_img = image + delta
    return out_img


def adjust_contrast(image, factor):
    img_mean = image.mean()
    out_img = (image - img_mean) * factor + img_mean
    return out_img


def mymovmen_3D(X, n):
    DataLength = X.shape[0]
    Y = np.zeros(X.shape)
    for i in range(DataLength):
        if i <= (DataLength-n):
            Y[i, :, :] = np.sum(X[i:i+n+1, :, :, ], axis=0) / n
        else:
            Y[i, :, :] = np.sum(X[i:, :, :, ], axis=0) / ((DataLength-i))
    return Y


def spatial_average(x1, x2, x3):
    y = cv2.GaussianBlur(x1.astype(np.float32),
                         (3, 3), cv2.BORDER_DEFAULT)
    y += cv2.GaussianBlur(x2.astype(np.float32),
                          (5, 5), cv2.BORDER_DEFAULT)
    y += cv2.GaussianBlur(x3.astype(np.float32),
                          (7, 7), cv2.BORDER_DEFAULT)
    return y


def read_frames(sequence_dir, target_dir, fnames):
    for frame in range(0, len(fnames)):
        fileName = str("t%03d.tif" % frame)
        mskName = str("man_seg%03d.tif" % frame)
        img = np.array(cv2.imread(str(sequence_dir + fileName), 0)) / 255
        img = img.astype(np.float32)
        img = cv2.medianBlur(img, 5)
        random_constrast_factor = np.random.rand() + 0.5
        random_brightness_delta = (np.random.rand() - 0.5) * 0.2 * img.max()
        img = adjust_brightness(img, random_brightness_delta)
        img = adjust_contrast(img, random_constrast_factor)
        img = (img - img.mean()) / (img.std())
        img = sigmoid(2 * img)
        img = cv2.resize(img, None, fx=0.2, fy=0.2,
                         interpolation=cv2.INTER_CUBIC)

        msk = np.array(cv2.imread(str(target_dir + mskName), 0))
        msk = cv2.resize(msk, None, fx=0.2, fy=0.2,
                         interpolation=cv2.INTER_CUBIC)
        msk[msk == 255] = 2
        msk[msk == 128] = 1

        nRow, nCol = img.shape
        if (frame == 0):
            im_stack = np.zeros((len(fnames), nRow, nCol))
            msk_stack = np.zeros((len(fnames), nRow, nCol))
            im_stack[frame, :, :] = img
            msk_stack[frame, :, :] = msk
        else:
            im_stack[frame, :, :] = img
#             if frame >= 2:
#                 im_stack[frame, :, :] = spatial_average(img, im_stack[frame-1, :, :],
#                                     im_stack[frame-2, :, :]) / 3
            msk_stack[frame, :, :] = msk
    im_stack = mymovmen_3D(im_stack, 5)  # contrast enhancement
    return im_stack, msk_stack, nRow, nCol


def readPixel_3D(Inp_mat, Out_mat, x0, y0):
    s = Inp_mat.shape[0]
    Objct_inp = np.array([Inp_mat[:, x0, y0],
                          Inp_mat[:, x0-1, y0-1],
                          Inp_mat[:, x0-1, y0],
                          Inp_mat[:, x0-1, y0+1],
                          Inp_mat[:, x0, y0-1],
                          Inp_mat[:, x0, y0+1],
                          Inp_mat[:, x0+1, y0-1],
                          Inp_mat[:, x0+1, y0],
                          Inp_mat[:, x0+1, y0+1]])
    Objct_inp = np.transpose(Objct_inp)
    Objct_out = np.reshape(Out_mat[:, x0, y0], (s, 1))
    return Objct_inp, Objct_out


def mymovmen_2D(X, n):
    DataLength = X.shape[0]
    Y = np.zeros(X.shape)
    for i in range(DataLength):
        if i <= (DataLength-n):
            Y[i, :] = np.sum(X[i:i+n+1, :, ], axis=0) / n
        else:
            Y[i, :] = np.sum(X[i:, :, ], axis=0) / ((DataLength-i))
    return Y


def postProcess(img, strel, er_strel, kernel, thr, minSize):
    pre_s = np.zeros(img.shape)
    pre_s[img >= thr] = 1
#     pre_s = grey_erosion(pre_s.astype(np.int32), structure=er_strel.astype(np.int8))
#     pre_s = grey_dilation(pre_s.astype(np.int32), structure=strel.astype(np.int8))
    cleanMask = np.float32(morphology.remove_small_objects(pre_s.astype(np.bool),
                                                           min_size=minSize, connectivity=8))
#     cleanMask = binary_fill_holes(cleanMask)
    cleanMask = cv2.filter2D(cleanMask.astype(np.float32), -1, kernel)
    cleanMask[cleanMask >= 0.5] = 1
    cleanMask[cleanMask < 0.5] = 0
    return cleanMask


# set path and read a single stream from the test directory
root_dir = '.~/Task3/RC/Data/NewData'
base_dir = str(root_dir)

sequence_dirs = ['/07/']
target_dirs = ['/07_GT/']
sequence_dir = str(base_dir) + str(sequence_dirs[0])
target_dir = str(base_dir) + str(target_dirs[0])
fnames = os.listdir(sequence_dir)
im_stack, msk_stack, nRow, nCol = read_frames(sequence_dir, target_dir, fnames)


with open('trainedESN_Encoding_02.pickle', 'rb') as handle:
    esn = pickle.load(handle)
print(esn)


# apply the trained RC to all pixels
esn.store_states_pred = True
pred_stack = np.zeros((len(fnames), 3, nRow, nCol))
pred_stack_class = np.zeros((len(fnames), nRow, nCol))

for x0 in range(1, nRow-1):
    if(x0 % 10 == 0):
        print('reading data from row', x0, 'out of', nRow, 'completed')
    for y0 in range(1, nCol-1):
        Objct_inp, Objct_out = readPixel_3D(im_stack, msk_stack, x0, y0)
        X_test = Objct_inp
        y_pred = esn.predict(X_test)
        pred_states = esn.states_pred_
        pred_states_smoothen = mymovmen_2D(pred_states, 50)
        y_pred = np.dot(np.hstack((pred_states_smoothen, X_test)),
                        np.transpose(esn.W_out_))
        pred_stack[:, :, x0, y0] = y_pred


# resize predictions and smoothen the predictions over time and space
pred_stack_resize = np.zeros((3, pred_stack.shape[0], 5 * nRow, 5 * nCol))
for channel in range(3):
    for frame in range(pred_stack.shape[0]):
        img = pred_stack[frame, channel, :, :]
        img_rs = cv2.resize(img, None, fx=5.0, fy=5.0,
                            interpolation=cv2.INTER_CUBIC)
        if frame >= 2:
            img_rs = spatial_average(img_rs, pred_stack_resize[channel, frame-1, :, :],
                                     pred_stack_resize[channel, frame-2, :, :]) / 2
        else:
            img_rs = cv2.GaussianBlur(img_rs.astype(np.float32),
                                      (3, 3), cv2.BORDER_DEFAULT)
        pred_stack_resize[channel, frame, :, :] = img_rs

for channel in range(3):
    pred_stack_resize[channel, :, :, :] = mymovmen_3D(pred_stack_resize[channel, :, :, :], 50)

# apply training class weights
pred_stack_resize[0, :, :, :] = 1.5 * pred_stack_resize[0, :, :]

# softamx and labels
y_pred_smax = softmax(pred_stack_resize, axis=0)
predict_class = np.argmax(y_pred_smax, axis=0)
predict_class = np.array(predict_class.tolist())

# create "Predictions" directory in the current path and save predictions
path = root_dir + str("/Predictions/")

# Post-processing parameters
thr = 0.5
kernel = np.ones((7, 7), np.float32) / 49
strel = np.zeros((4, 4))
er_strel = np.zeros((2, 2))

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

for frame in range(predict_class.shape[0]):
    # Filename
    filename = str("t%03d.png" % frame)
    prdct_cell = np.zeros((5 * nRow, 5 * nCol))
    prdct_bead = np.zeros((5 * nRow, 5 * nCol))
    prdct_background = np.zeros((5 * nRow, 5 * nCol))

    prdct_cell[predict_class[frame, :, :] == 0] = 1
    prdct_bead[predict_class[frame, :, :] == 1] = 1
    prdct_background[predict_class[frame, :, :] == 2] = 1

    prdct_cell = postProcess(prdct_cell, strel, er_strel, kernel, thr, 300)
    prdct_cell = binary_fill_holes(prdct_cell)
    prdct_bead = postProcess(prdct_bead, strel, er_strel, kernel, thr, 200)
    prdct_bead[prdct_cell == 1] = 0
    prdct_bead = binary_fill_holes(prdct_bead)
    prdct_background = postProcess(prdct_background, strel, er_strel, kernel, thr, 200)
    img = 255 * prdct_cell
    img[prdct_bead > 0] = 128
    pair_img = 255 * np.dstack((y_pred_smax[0, frame, :, :],
                                y_pred_smax[1, frame, :, :],
                                y_pred_smax[2, frame, :, :]))

    # Saving the image
    cv2.imwrite(str(path_softmax)+str(filename), pair_img)
    cv2.imwrite(str(path_mask)+str(filename), img)
