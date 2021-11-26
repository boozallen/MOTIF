#!/usr/bin/env python3
import os
import numpy as np
import argparse
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from EMBER import read_vectorized_features, train_model

RANDOM_STATE = 42

# Check if the input is a valid directory
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train LightGBM  model")
    parser.add_argument("dataset_dir", type=dir_path, help="Path to directory containing MOTIF dataset")
    args = parser.parse_args()

    # Load feature vectors
    X, y = read_vectorized_features(args.dataset_dir)

    # Keep only families with more than one sample
    label_counts = Counter(y)
    keep_labels = set([label for label, count in label_counts.items() if label_counts[label] > 1])
    keep_idxs = [idx for idx, label in enumerate(y) if label in keep_labels]
    X_keep, y_keep = X[keep_idxs], y[keep_idxs]

    # Get number of remaining malware families
    num_class = len(keep_labels)

    # Re-map labels to be in range [0, num_class)
    label_map = dict(zip(sorted(keep_labels), np.arange(num_class, dtype=np.float32)))
    y_keep = np.array([label_map[label] for label in y_keep])

    # Perform 5-fold cross validation
    accuracies = []
    skf = StratifiedKFold(random_state=42, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(skf.split(X_keep, y_keep)):

        # Get train and test sets for fold
        X_train, y_train = X_keep[train_idx], y_keep[train_idx]
        X_test, y_test = X_keep[test_idx], y_keep[test_idx]

        print("Training model on fold {}".format(i+1))
        clf = train_model(X_train, y_train, params={"num_class": num_class, "verbose": 0})
        predictions = np.argmax(clf.predict(X_test), axis=1)
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
    print("Accuracy scores: {}".format(accuracies))
