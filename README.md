# Jurkat_cell_segmentation

The scripts in this repository are suplamentary to our paper "Spatio-temporal feature learning with reservoir computing for T-cell segmentation in live-cell 
Ca<sup>2+</sup> fluorescence microscopy" published in Scientific Reports. If this code is used please cite the paper:

@article{hadaeghi2021, title={Spatio-temporal feature learning with reservoir computing for T-cell segmentation in live-cell 
Ca<sup>2+</sup> fluorescence microscopy}, author={Fatemeh Hadaeghi, Bj\"{o}rn-Philipp Diercks, Daniel Schetelig, Fabrizio Damicelli, Insa M.A. Wolf, Ren\'{e} Werner}, journal={Scientific Reports}, volume={11}, pages={8233}, year={2021}}

# Prerequisites
This project describes segmentation approaches tailored to the requirements of live-cell Ca<sup>2+</sup> microscopy and Ca<sup>2+</sup> signaling analysis in T-cells. We focused on three segmentation scenarios: 
- Task 1: single object segmentation,
- Task 2: T-cell and bead segmentation and differentiation exploiting two-channel information,
- Task 3: T-cell/bead segmentation and differentiation in single-channel recordings.

We trained Reservoir Computing models, standard U-Net and convolutional long short-term memory (LSTM) models (https://github.com/arbellea/LSTM-UNet). The scripts are writen in Python 3 and make use of tensorflow 2.0.0a0., fastai (https://www.fast.ai/), and echoes (https://github.com/fabridamicelli/echoes). Please see the requierments.txt files for all prerequisits in different practices.

# Data
The training scripts was tailored for domestic live-cell Ca<sup>2+</sup> fluorescence microscopy data. Dataset corresponding to each task are available in corresponding directories.

## Private Data:
If you would like to train models on your private data, please follow the following formats to prepare your data:
--root_dir : Root directory of each sequence, example: '~/CaImaging/Task1/Green/Training/

--seq : Sequence number (two digit) , example: '01' or '02'

--frame_file_template : Template for image (frame) sequences, example: '01/t{:03d}.tif' where {:03d} will be replaced by a three digit number 000,001,...

-- mask_file_template : TemplateTemplate for reference sequences segmentation , example: '01_GT/man_seg{:03d}.tif' where {:03d} will be replaced by a three digit number 000,001,...

# Train and Inference
## Train and inference on private data
In each case the training and inference scripts are Train.py and Inference.py, respectively. On private data, you may need to adjust reservoir hyperparameters via cross-validation before training. 

## Inference on trained models
In each case, the trained RC models are available in corresponding directories. You only need to adjust --root_dir and run 

`python Inference.py`.

A "Prediction" directory with predicted masks will be created in the same directory.

# Baseline models
We compared the RC performance with a standard U-Net and a convolutional long short-term memory (LSTM) model. The trained models are available in the following links and corresponding scripts would be shared upon private request (f.hadaeghi@uke.de):
- Task 1: [U-Net](https://drive.google.com/file/d/19KGWGYXnjp16-5cD2dARP5PMB3-MabPR/view?usp=sharing), [U-Net-LSTM](https://drive.google.com/file/d/16mfVGM3aDX5bUs__ycffjJItxqETPS6E/view?usp=sharing)
- Task 2: [U-Net](https://drive.google.com/file/d/1KJQ3YXf_ylRq_jbaSqp9yz_5K7FAguCa/view?usp=sharing), [U-Net-LSTM](https://drive.google.com/file/d/1sM_gUaJn3zjfnjpF2dvaFFHUz5BdmsG3/view?usp=sharing)
- Task 3: [U-Net-LSTM](https://drive.google.com/file/d/1YbljjUloOBDkR85MdxjfiIVy8K8hGbgI/view?usp=sharing)

