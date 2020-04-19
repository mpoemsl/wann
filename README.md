# WANN
[Weight-Agnostic Neural Networks](https://weightagnostic.github.io/) Implementation.

A Course Project for "Implementing ANNs with TensorFlow" in WS 2019/20 at Osnabrück University by mpoemsl and jkaltenborn.

The re-implementation is loosely based on:
[Gaier, A. and D. Ha (2019). Weight agnostic neural networks.](https://arxiv.org/pdf/1906.04358)

## Overview 

This projects consist of scripts to train, test and visualize WANNs as well as ANNs on the classification task [MNIST Handwritten Digit Database](http://yann.lecun.com/exdb/mnist/) and the regression task [Forest Fires](http://archive.ics.uci.edu/ml/datasets/Forest+Fires). 

The Python scripts `train_wann.py`, `test_wann.py` and `visualize_wann.py` can be used to perform WANN experiments. The Python script `run_ann.py` can be used to perform ANN experiments. To learn something about their parameters, run `python [SCRIPT].py -h`. All other Python scripts are supporting. `plots` stores created visuals and `experiments` stores created experiments. Once properly installed, `run_experiments.sh` can be used to run all experiments. `wann_report.pdf` serves as documentation.

## Installation 

This project runs on Python 3 with various supporting packages, which are the easiest to install with `pip install -r requirements.txt`. In particular, make sure to install `tensorflow`, `numpy` and `tfds-nightly`, which is needed to load the Forest Fires dataset.

## Usage

To train a WANN, run `python train_wann.py DATASET`, where `DATASET` is one of {forestfires, mnist}. This will create a corresponding experiment folder in `experiments`.

* Run `python test_wann.py EXP_FOLDER` to evaluate the best individuums of each generation. 
* Run `python visualize_wann.py EXP_FOLDER` to visualize statistics of this experiment and store them in `plots`.
* Run `python run_ann.py DATASET` to train, test and visualize a comparable ANN

Or run `bash run_experiments.sh` to do all of the above.
