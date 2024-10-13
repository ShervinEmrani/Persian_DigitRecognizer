# Handwritten Digit Recognition on HODA Dataset

This project implements two classifiers for handwritten digit recognition on the HODA dataset. The classifiers used are:
1. Random Forest
2. K-Nearest Neighbors (KNN)

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Classifiers](#classifiers)

## Project Overview

The goal of this project is to classify handwritten digits using two machine learning classifiers: Random Forest and KNN. The dataset used is the **HODA dataset**, which contains Persian/Arabic handwritten digits.

The classifiers are evaluated on test data, and the performance is measured using accuracy and confusion matrix.

## Dataset

The HODA dataset is a Persian/Arabic handwritten digit dataset, and it is processed in this project to have a fixed image size of 32x32 pixels. The dataset is divided into:

- **Training set**: 60,000 images
- **Test set**: 20,000 images


## Classifiers

### 1. Random Forest

A Random Forest classifier is trained on the HODA dataset. The model is created with `n_estimators=30` (number of decision trees). It works by constructing multiple decision trees during training, and the output is the class that is the mode of the classes (for classification) of the individual trees.

After training, the classifier is evaluated on the test data, and the accuracy is computed.

#### Key Parameters:
- `n_estimators=30`: Number of trees in the forest.
- `criterion='gini'`: The function to measure the quality of a split.
- `max_depth=None`: The maximum depth of the tree. If `None`, nodes are expanded until all leaves are pure or until all leaves contain less than `min_samples_split` samples.

#### Evaluation:
The model is tested on the HODA test set and evaluated using metrics like accuracy and a confusion matrix, which measures the performance of the classification.

### 2. K-Nearest Neighbors (KNN)

The K-Nearest Neighbors (KNN) classifier is implemented using `scikit-learn`. This model classifies data points based on the `k` closest training examples in the feature space. The distance between data points is typically calculated using Euclidean distance.

#### Key Parameters:
- `n_neighbors=5`: Number of neighbors to use for classification.
- `metric='minkowski'`: The distance metric used for the tree.
- `weights='uniform'`: Uniform weights mean all points in each neighborhood are weighted equally.

#### Evaluation:
After training the KNN model on the HODA dataset, the classifier is evaluated on the test set. The performance is measured through accuracy scores and confusion matrix visualization, similar to the Random Forest model.
