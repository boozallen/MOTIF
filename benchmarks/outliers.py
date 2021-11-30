#!/usr/bin/env python3
import os
import numpy as np
import argparse
from collections import Counter
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support
from ember import read_vectorized_features

RANDOM_STATE = 42

# Check if the input is a valid directory
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train outlier detection models")
    parser.add_argument("dataset_dir", type=dir_path, help="Path to directory containing MOTIF dataset")
    args = parser.parse_args()

    # Load feature vectors
    X, y = read_vectorized_features(args.dataset_dir, subset="train")

    # Malware families with only one "representative" will be outliers
    label_counts = Counter(y)
    outlier_labels = set([label for label, count in label_counts.items() if label_counts[label] == 1])
    inlier_labels = set(y) - outlier_labels

    # Separate outliers and inliers
    outlier_idxs = [idx for idx, label in enumerate(y) if label in outlier_labels]
    inlier_idxs = [idx for idx, label in enumerate(y) if label in inlier_labels]
    X_outliers, y_outliers = X[outlier_idxs], y[outlier_idxs]
    X_inliers, y_inliers = X[inlier_idxs], y[inlier_idxs]

    # Train/Test split, add outliers to test set
    X_train, X_test, y_train, y_test = train_test_split(X_inliers, y_inliers, test_size=0.1, random_state=42)
    X_test = np.concatenate((X_test, X_outliers))
    y_test = np.array([1 for _ in range(len(y_test))] + [-1 for _ in range(len(y_outliers))])
    X_test, y_test = shuffle(X_test, y_test)

    # Get predictions from each model
    clfs = {
        "Isolation Forest": IsolationForest(random_state=RANDOM_STATE),
        "Local Outlier Factor": LocalOutlierFactor(novelty=True),
        "One Class SVM": OneClassSVM()
    }    
    for clf_name, clf in clfs.items():
        clf.fit(X_train)
        predictions = clf.predict(X_test)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, average="binary", pos_label=-1)
        print("Precision, Recall, F1 of {}: {}, {}, {}".format(clf_name, precision, recall, f1))
