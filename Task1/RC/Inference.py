#!/usr/bin/env python
# coding: utf-8

""" Loads a trained reservoir computing model and segments single objects
in a stream of Ca Imaging data. """

__author__ = 'f.hadaeghi@uke.de'

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import grey_dilation, grey_erosion
from scipy.ndimage import binary_fill_holes
from skimage import morphology
# in case of cross-validation
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split

from echoes import ESNRegressor
from echoes.plotting import set_mystyle
from echoes.reservoir._leaky_numba import harvest_states
from numba import njit

import pickle


# load RC model (Green)
with open('trainedESN_Green.pickle', 'rb') as handle:
    esn = pickle.load(handle)

# load RC model (Red)
# with open('trainedESN_Red.pickle', 'rb') as handle:
#     esn = pickle.load(handle)
print(esn)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum(axis=0) 


def PostProcess(img, strel, er_strel, kernel, thr):
    pre_s = np.zeros(img.shape)
    pre_s[img >= thr] = 1
#     pre_s = grey_erosion(pre_s.astype(np.int32), structure=er_strel.astype(np.int8))
#     pre_s = grey_dilation(dilation.astype(np.int32), structure=strel.astype(np.int8))
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
    numFrame = int(gt.shape[0]/(nRow*nCol))
    start_pixel = np.mod(gt.shape[0], nRow*nCol)
    tg_output = np.reshape(gt[-numFrame*nRow*nCol-1:-1], (numFrame, nRow, nCol)).astype(np.float64)
    pr_output = np.reshape(pred[-numFrame*nRow*nCol-1:-1],(numFrame,nRow,nCol)).astype(np.float64)
    return tg_output, pr_output


def prediction_resize(tg, prdct, resize_factor):
    [nFrame, nRows, nCols] = tg.shape
    tg_resize = np.zeros((nFrame, nRows*resize_factor, nCols*resize_factor))
    prdct_resize = np.zeros((nFrame, nRows*resize_factor, nCols*resize_factor))    
    for frame in range(nFrame):
        tg_resize[frame, :, :] = cv2.resize(tg[frame,:,:], None, fx=resize_factor, 
                                            fy=resize_factor, interpolation=cv2.INTER_CUBIC)
        prdct_resize[frame, :, :] = cv2.resize(prdct[frame,:,:], None, fx=resize_factor, 
                                               fy=resize_factor, interpolation=cv2.INTER_CUBIC)
    prdct_resize = mymovmen(prdct_resize, 5)  # optional temporal averaging to improve temporal consistency
#     prdct_resize = (prdct_resize - prdct_resize.mean()) / (prdct_resize.std())
    prdct_resize = (prdct_resize - prdct_resize.min()) / (prdct_resize.max()-prdct_resize.min())

    tg_resize[tg_resize >= 0.5] = 1
    tg_resize[tg_resize < 0.5] = 0
    return tg_resize, prdct_resize


def stack_post_process(prdct, strel, er_strel, kernel, thr):
    binary_prdct = np.zeros(prdct.shape)
    for frame in range(prdct.shape[0]):    
        binary_prdct[frame,:,:] = PostProcess(prdct[frame,:,:], strel, er_strel, kernel, thr)
    return binary_prdct   


# set the paths
root_dir = '/home/fatemeh/Documents/Projects/CaImaging/GitHub_Repo/Task1/'
# root_dir = '.~/Task1/'  # Green emission
# root_dir = '.~/Task1/'  # Red emission

base_dir = str(root_dir + '/RC/SampleData/')
sequence_dir = str(base_dir + '22/')
target_dir = str(root_dir + '/RC/SampleData/' + 'GT_22/')
fnames = os.listdir(sequence_dir)
for frame in range(0, len(fnames)):
    fileName = str("t%03d.tif" % frame)
    mskName = str("man_seg%03d.tif" % frame)
    if (frame == 0):
        img = np.array(cv2.imread(str(sequence_dir + fileName), 0)) / 255
        img = img.astype(np.float32)
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)                 
        img = cv2.medianBlur(img, 5)
        img = (img - img.mean()) / (img.std())
        img = sigmoid(img)
        ColIm = np.reshape(img, (-1, 1))
        RowIm = np.reshape(np.transpose(img), (-1, 1))
        ImgVector = np.hstack((ColIm, RowIm))
        nRow, nCol = img.shape

        tmp_msk = np.array(cv2.imread(str(target_dir + mskName),0)) / 255
        tmp_msk = cv2.resize(tmp_msk, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        tmp_msk[tmp_msk >= 0.5] = 1
        tmp_msk[tmp_msk < 0.5] = 0
        msk = np.reshape(tmp_msk, (-1, 1))

    img = np.array(cv2.imread(str(sequence_dir + fileName),0)) / 255
    img = img.astype(np.float32)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.medianBlur(img, 5)
    img = (img - img.mean()) / (img.std())
    img = sigmoid(img)

    ColIm = np.reshape(img, (-1, 1))
    RowIm = np.reshape(np.transpose(img), (-1, 1))
    tmpVector = np.hstack((ColIm, RowIm))
    ImgVector = np.vstack((ImgVector, tmpVector))
    tmp_msk = np.array(cv2.imread(str(target_dir + mskName), 0)) / 255
    tmp_msk = cv2.resize(tmp_msk, None,fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    tmp_msk[tmp_msk >= 0.5] = 1
    tmp_msk[tmp_msk < 0.5] = 0

    tmp_msk_vec = np.reshape(tmp_msk, (-1, 1))
    msk = np.vstack((msk, tmp_msk_vec))
# print(msk.shape, ImgVector.shape)


AugmentedIm = np.hstack((ImgVector, np.hstack((np.roll(ImgVector, 1, axis=0), np.roll(ImgVector, -1, axis=0)))))

inputs = AugmentedIm.astype(np.float64)
outputs = msk.astype(np.float64)

X_test = inputs
y_test = outputs

# one option to improve segmention results is to repeat the input time-series
# and then throw away the first half of predicted time-series, for instance:
# X_test = np.tile(inputs, (2,1))
# y_test = np.tile(outputs, (2,1))


y_pred = esn.predict(X_test)
# In case the time-series is repeated once 
# index = np.arange(int(y_pred.shape[0]/2)).reshape(int(y_pred.shape[0]/2),1)
# y_pred = np.delete(y_pred, index)
# y_test = np.delete(y_test, index)
# print (y_pred.shape[0])

kernel = np.ones((3,3),np.float32) / 9
strel = np.zeros((3, 3))
er_strel = np.zeros((2, 2))

# reconstruct and resize frames
tg_output, pr_output = frame_reconstruct(y_test,y_pred,nRow,nCol)
tg_resize, prdct_resize = prediction_resize(tg_output, pr_output, 2)
prdct_resize_binary = stack_post_process(prdct_resize, strel, er_strel, kernel, 0.45)

# Visulaize frames
#frame = 30
#trgt =tg_resize[frame,:,:]
#prdct = prdct_resize_binary[frame,:,:]
#pair_img = np.dstack((trgt,prdct,trgt))
#imgplot = plt.imshow(pair_img, cmap="gray")
#plt.xticks([]), plt.yticks([])

#save predictions

path = root_dir + str("RC/Predictions/")

try:
    os.mkdir(path)

except OSError:
    print("Creation of the directory %s failed" % path)
else:
    print("Successfully created the directory %s " % path)

for frame in range(prdct_resize_binary.shape[0]):
    # Filename
    filename = str("t%03d.png" % frame)                   
    # Saving the image
    cv2.imwrite(str(path)+str(filename), 255*prdct_resize_binary[frame,:,:])
