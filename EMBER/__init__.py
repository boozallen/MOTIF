#!/usr/bin/env python3
import os
import json
import tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb
import multiprocessing
from .features import PEFeatureExtractor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, make_scorer)


def raw_feature_iterator(file_paths):
    """
    Yield raw feature strings from the inputed file paths
    """
    for path in file_paths:
        with open(path, "r") as fin:
            for line in fin:
                yield line


def vectorize(irow, raw_features_string, X_path, y_path, extractor, nrows):
    """
    Vectorize a single sample of raw features and write to a large numpy file
    """
    raw_features = json.loads(raw_features_string)
    feature_vector = extractor.process_raw_features(raw_features)

    y = np.memmap(y_path, dtype=np.float32, mode="r+", shape=nrows)
    y[irow] = raw_features["label"]

    X = np.memmap(X_path, dtype=np.float32, mode="r+", shape=(nrows, extractor.dim))
    X[irow] = feature_vector


def vectorize_unpack(args):
    """
    Pass through function for unpacking vectorize arguments
    """
    return vectorize(*args)


def vectorize_subset(X_path, y_path, raw_feature_paths, extractor, nrows):
    """
    Vectorize a subset of data and write it to disk
    """
    # Create space on disk to write features to
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
    y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=nrows)
    del X, y

    # Distribute the vectorization work
    pool = multiprocessing.Pool()
    argument_iterator = ((irow, raw_features_string, X_path, y_path, extractor, nrows)
                         for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths)))
    for _ in tqdm.tqdm(pool.imap_unordered(vectorize_unpack, argument_iterator), total=nrows):
        pass


def create_vectorized_features(data_dir, feature_version=2):
    """
    Create feature vectors from raw features and write them to disk
    """
    extractor = PEFeatureExtractor(feature_version)

    print("Vectorizing training set")
    X_path = os.path.join(data_dir, "X.dat")
    y_path = os.path.join(data_dir, "y.dat")
    raw_feature_paths = [os.path.join(data_dir, "motif_dataset.jsonl")]
    nrows = sum([1 for fp in raw_feature_paths for line in open(fp)])
    vectorize_subset(X_path, y_path, raw_feature_paths, extractor, nrows)


def read_vectorized_features(data_dir, subset=None, feature_version=2):
    """
    Read vectorized features into memory mapped numpy arrays
    """
    extractor = PEFeatureExtractor(feature_version)
    ndim = extractor.dim

    X_path = os.path.join(data_dir, "X.dat")
    y_path = os.path.join(data_dir, "y.dat")
    y = np.memmap(y_path, dtype=np.float32, mode="r")
    N = y.shape[0]
    X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(N, ndim))

    return X, y


def train_model(X, y, params={}, feature_version=2):
    """
    Train the LightGBM model from the EMBER dataset from the vectorized features
    """
    # update params
    params.update({"application": "multiclass"})

    # Filter unlabeled data
    train_rows = (y != -1)

    # Train
    lgbm_dataset = lgb.Dataset(X[train_rows], y[train_rows])
    lgbm_model = lgb.train(params, lgbm_dataset)

    return lgbm_model


def predict_sample(lgbm_model, file_data, feature_version=2):
    """
    Predict a PE file with an LightGBM model
    """
    extractor = PEFeatureExtractor(feature_version)
    features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
    return lgbm_model.predict([features])[0]
