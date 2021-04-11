#!/usr/bin/env python
# coding: utf-8

""" Trains a reservoir computing model on streams of Ca Image data
measured with Fluo-4 (green) emission to segmen/classify
existing objects (T-cells and beads) (Encoding scheme 2). """

__author__ = 'f.hadaeghi@uke.de'

# imports
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import grey_dilation, affine_transform
from scipy.ndimage import grey_erosion, binary_fill_holes
from skimage import morphology
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

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
            Y[i, :] = np.sum(X[i:i+n+1, :, ], axis=0) / n
        else:
            Y[i, :] = np.sum(X[i:, :, ], axis=0) / ((DataLength-i))
    return Y


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


def sample_pixels(Inp_mat, Out_mat, numFrame, numSamples, sampleFrame):
    Data = np.zeros((numSamples*numFrame, 10, 3))

    for lbl in range(3):
        x_idx, y_idx = np.where(Out_mat[sampleFrame, :, :] == lbl)
        Data_background_inp = np.zeros((numFrame, 9))
        Data_background_out = np.zeros((numFrame, 1))
        idx = 0
        for i in range(0, 4 * numSamples):
            if(idx >= numSamples-1):
                break
            else:
                # randomly choose any element in the x_idx list
                rand_idx = np.random.choice(x_idx)
                x0 = x_idx[rand_idx]
                y0 = y_idx[rand_idx]
                if((x0 > 0) & (y0 > 0) & (x0 < nRow-1) & (y0 < nCol-1)):
                    if(i == 0):
                        Data_background_inp, Data_background_out = readPixel_3D(Inp_mat, Out_mat, x0, y0)
                    else:
                        idx += 1
                        data_inp, data_out = readPixel_3D(Inp_mat, Out_mat, x0, y0)
                        Data_background_inp = np.vstack((Data_background_inp, data_inp))
                        Data_background_out = np.vstack((Data_background_out, data_out))

        Data[:, :, lbl] = np.hstack((Data_background_inp, lbl * np.ones((Data_background_inp.shape[0], 1))))
    return Data


sequence_dirs = ['01/', '02/', '03/', '04/', '05/']
target_dirs = ['01_GT/', '02_GT/', '03_GT/', '04_GT/', '05_GT/']

TrainData = np.zeros((1, 10))

for seq in range(len(sequence_dirs)):
    sequence_dir = str(base_dir) + str(sequence_dirs[seq])
    target_dir = str(base_dir) + str(target_dirs[seq])
    print(target_dir)
    fnames = os.listdir(sequence_dir)
    im_stack, msk_stack, nRow, nCol = read_frames(sequence_dir, target_dir, fnames)
    Data = sample_pixels(im_stack, msk_stack, len(fnames), 1000, 150)
    tmp = np.vstack((np.vstack((Data[:, :, 0], Data[:, :, 1])), Data[:, :, 2]))
    TrainData = np.vstack((TrainData, tmp))
    print('sequence', sequence_dirs[seq], 'completed')


# Split outputs
OutVect = TrainData[:, 9].astype(int)

out_1 = np.zeros((OutVect.shape[0], 1))
out_2 = np.zeros((OutVect.shape[0], 1))
out_3 = np.zeros((OutVect.shape[0], 1))

out_1[OutVect == 2] = 1  # cell
out_2[OutVect == 1] = 1  # bead
out_3[OutVect == 0] = 1  # background

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


X_train = TrainData[0::2, 0:9]
X_test = TrainData[0::2, 0:9]
y_train = OutputVect[0::2]
y_test = OutputVect[0::2]


class_weights = [1.5, 1.0, 1.0]
y_train = class_weights * y_train
esn = ESNRegressor(
    n_reservoir=500,
    spectral_radius=0.99,
    leak_rate=.5,
    noise=0.0001,
    n_transient=0,
    input_scaling=1.0,
    input_shift=0.0,
    sparsity=0.98,
    bias=0.2,
    activation_out=mysigmoid,
    activation=myrelu,
    regression_method="ridge",
    ridge_alpha=0.000001,
    random_state=42,
    ridge_normalize=False,
    ridge_fit_intercept=False,
    store_states_train=False,
    store_states_pred=False,
)

esn.fit(X_train, y_train)
# For cross-validation
# y_pred = esn.predict(X_test)
with open('trainedESN_Encoding_02.pickle', 'wb') as handle:
    pickle.dump(esn, handle, protocol=pickle.HIGHEST_PROTOCOL)
