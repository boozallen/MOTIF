#!/usr/bin/env python3
import os
import json
import random
import argparse
from collections import Counter, deque
import numpy as np
from tqdm import tqdm
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from MalConv2.MalConv import MalConv
from MalConv2.binaryLoader import BinaryDataset, pad_collate_func

# Check if the input is a valid directory
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

# Check if the input is a valid file path
def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a MalConv model")
    parser.add_argument("--filter_size", type=int, default=512, help="How wide should the filter be")
    parser.add_argument("--filter_stride", type=int, default=512, help="Filter Stride")
    parser.add_argument("--embd_size", type=int, default=8, help="Size of embedding layer")
    parser.add_argument("--num_channels", type=int, default=128, help="Total number of channels in output")
    parser.add_argument("--epochs", type=int, default=10, help="How many training epochs to perform")
    parser.add_argument("--non-neg", type=bool, default=False, help="Should non-negative training be used")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size during training")
    # Default is set at 16 MB! 
    parser.add_argument("--max_len", type=int, default=16000000, help="Maximum length of input file in bytes, at which point files will be truncated")
    parser.add_argument("--gpus", nargs='+', type=int)
    parser.add_argument("mal_dir", type=dir_path, help="Path to directory containing malware files")
    parser.add_argument("label_path", type=file_path, help="Path to .jsonl file containing labeled md5s")
    args = parser.parse_args()

    GPUS = args.gpus
    NON_NEG = args.non_neg
    EMBD_SIZE = args.embd_size
    FILTER_SIZE = args.filter_size
    FILTER_STRIDE = args.filter_stride
    NUM_CHANNELS = args.num_channels
    EPOCHS = args.epochs
    MAX_FILE_LEN = args.max_len
    BATCH_SIZE = args.batch_size
    RANDOM_STATE = 42

    # Map md5s to labels
    file_names = {}
    md5_labels = []
    labels = []
    with open(args.label_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            md5_labels.append((entry["md5"], entry["label"]))
            labels.append(entry["label"])
    labels = np.array(labels, dtype=np.int64)

    # Keep only families with more than one sample
    label_counts = Counter(labels)
    keep_labels = set([label for label, count in label_counts.items() if label_counts[label] > 1])
    keep_idxs = [idx for idx, label in enumerate(labels) if label in keep_labels]
    keep_md5_labels = [md5_labels[i] for i in keep_idxs]
    y_keep = labels[keep_idxs]

    # Get number of remaining malware families
    num_class = len(keep_labels)

    # Re-map labels to be in range [0, num_class)
    label_map = dict(zip(sorted(keep_labels), np.arange(num_class, dtype=np.int64)))
    y_keep = np.array([label_map[label] for label in y_keep])
    keep_md5_labels = [(md5, label_map[label]) for md5, label in keep_md5_labels]

    # Initialize dataset
    dataset = BinaryDataset(args.mal_dir, keep_md5_labels, max_len=MAX_FILE_LEN)
    loader_threads = max(multiprocessing.cpu_count()-4, multiprocessing.cpu_count()//2+1)

    # Determine device
    if GPUS is None:
        device_str = "cuda:0"
    else:
        if GPUS[0] < 0:
            device_str = "cpu"
        else:
            device_str = "cuda:{}".format(GPUS[0])
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("Using device ", device)

    # Stratified k-fold cross validation
    fold_stats = []
    skf = StratifiedKFold(random_state=RANDOM_STATE, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), y_keep)):

        # Initialize train and test DataLoaders
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=loader_threads, collate_fn=pad_collate_func,
                                  sampler=SubsetRandomSampler(train_idx))
        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=loader_threads, collate_fn=pad_collate_func,
                                 sampler=SubsetRandomSampler(test_idx))

        # Initialize MalConv model
        model = MalConv(out_size=num_class, channels=NUM_CHANNELS, window_size=FILTER_SIZE, stride=FILTER_STRIDE, embd_size=EMBD_SIZE).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        # Train for each epoch
        model.train()
        print("Training model on fold {}".format(i+1))
        for epoch in tqdm(range(EPOCHS)):
            preds = []
            truths = []
            running_loss = 0.0
            train_correct = 0
            train_total = 0
            epoch_stats = {"epoch": epoch}
            for inputs, labels in tqdm(train_loader):
                labels = labels.to(device)
                optimizer.zero_grad()                
                outputs, _, _ = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if NON_NEG:
                    for p in model.parameters():
                        p.data.clamp_(0)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                with torch.no_grad():
                    preds.extend(F.softmax(outputs, dim=-1).data[:,1].detach().cpu().numpy().ravel())
                    truths.extend(labels.detach().cpu().numpy().ravel())
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                epoch_stats["train_acc"] = train_correct*1.0/train_total

        # Evaluate using test set
        model.eval()
        print("Testing model on fold {}".format(i+1))
        eval_train_correct = 0
        eval_train_total = 0
        preds = []
        truths = []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _, _ = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                preds.extend(F.softmax(outputs, dim=-1).data[:,1].detach().cpu().numpy().ravel())
                truths.extend(labels.detach().cpu().numpy().ravel())            
                eval_train_total += labels.size(0)
                eval_train_correct += (predicted == labels).sum().item()
                epoch_stats["test_acc"] = eval_train_correct*1.0/eval_train_total

        # Save stats for current epoch
        fold_stats.append(epoch_stats)

    for i, stats in enumerate(fold_stats):
        print("Fold {} Testing Accuracy: {}".format(i+1, stats["test_acc"]))
