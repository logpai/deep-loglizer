#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../")
from deeploglizer.models import LSTM
from deeploglizer.common.dataloader import load_HDFS, log_dataset
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.utils import seed_everything, set_device
from torch.utils.data import DataLoader

random_seed = 42
device = 0

sequential_partition = True
test_ratio = 0.2
window_size = 10
stride = 1

topk = 5
batch_size = 512
epoches = 2
learning_rate = 1.0e-3

hidden_size = 32
num_directions = 2

max_token_len = 50  # max #token for each event [semantic only]
min_token_count = 1  # min # occurrence of token for each event [semantic only]


log_file = "../data/HDFS/HDFS_100k.log_structured.csv"  # The structured log file
label_file = "../data/HDFS/anomaly_label.csv"  # The anomaly label file

if __name__ == "__main__":
    seed_everything(random_seed)

    session_train, session_test = load_HDFS(
        log_file,
        label_file=label_file,
    )

    ext = FeatureExtractor(
        label_types="next_log",  # "none", "next_log", "anomaly"
        feature_types=["sequentials"],
        window_types="sliding",
        window_size=window_size,
        stride=stride,
        max_token_len=max_token_len,
        min_token_count=min_token_count,
    )

    ext.fit_transform(session_train)
    ext.transform(session_test, datatype="test")

    num_labels = ext.meta_data["num_labels"]

    dataset_train = log_dataset(session_train)
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    dataset_test = log_dataset(session_test)
    dataloader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    model = LSTM(num_labels=num_labels, topk=topk, device=device)
    model.fit(dataloader_train, learning_rate=learning_rate)
    model.evaluate(dataloader_test)
