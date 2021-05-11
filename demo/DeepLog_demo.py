#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

sys.path.append("../")
from deeploglizer.models import LSTM
from deeploglizer.common.dataloader import (
    load_HDFS,
    log_dataset,
    load_HDFS_id,
    load_HDFS_semantic,
)
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.utils import seed_everything, set_device
from torch.utils.data import DataLoader

from IPython import embed

random_seed = 42
device = 0

label_type = "next_log"
feature_type = "sequentials"  # "sequentials", "semantics", "quantitatives"
window_size = 10
stride = 1

topk = 10
batch_size = 1024
epoches = 50
learning_rate = 1.0e-2
use_tfidf = False

hidden_size = 200
num_directions = 1
embedding_dim = 16

max_token_len = 50  # max #token for each event [semantic only]
min_token_count = 1  # min # occurrence of token for each event [semantic only]
pretrain_path = None
# pretrain_path = "../data/pretrain/wiki-news-300d-1M.vec"

deduplicate_windows = False
cache = False

log_file = "../data/HDFS/HDFS.log_groundtruth.csv"  # The structured log file
if not os.path.isfile(log_file):
    log_file = "../data/HDFS/HDFS_100k.log_structured.csv"  # The structured log file
label_file = "../data/HDFS/anomaly_label.csv"  # The anomaly label file

if __name__ == "__main__":
    seed_everything(random_seed)

    session_train, session_test = load_HDFS(
        log_file=log_file,
        label_file=label_file,
        test_ratio=0.8,
        sequential_partition=False,
        random_seed=42,
    )

    # session_train, session_test = load_HDFS_semantic("../data/HDFS_semantic")

    ext = FeatureExtractor(
        label_type=label_type,  # "none", "next_log", "anomaly"
        feature_type=feature_type,  # "sequentials", "semantics", "quantitatives"
        window_type="sliding",
        window_size=window_size,
        stride=stride,
        max_token_len=max_token_len,
        min_token_count=min_token_count,
        pretrain_path=pretrain_path,
        use_tfidf=use_tfidf,
        deduplicate_windows=deduplicate_windows,
        cache=cache,
    )

    session_train = ext.fit_transform(session_train)
    session_test = ext.transform(session_test, datatype="test")

    dataset_train = log_dataset(session_train, feature_type=feature_type)
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    dataset_test = log_dataset(session_test, feature_type=feature_type)
    dataloader_test = DataLoader(
        dataset_test, batch_size=4096, shuffle=False, pin_memory=True
    )

    model = LSTM(
        meta_data=ext.meta_data,
        hidden_size=hidden_size,
        num_directions=num_directions,
        embedding_dim=embedding_dim,
        feature_type=feature_type,
        label_type=label_type,
        use_tfidf=use_tfidf,
        topk=topk,
        device=device,
    )
    model.fit(
        dataloader_train,
        test_loader=dataloader_test,
        epoches=epoches,
        learning_rate=learning_rate,
    )

    # print("Evaluating train:")
    # eval_results = model.evaluate(dataloader_train, "train")
    # print(eval_results)

    # print("Evaluating test:")
    # eval_results = model.evaluate(dataloader_test)
    # print(eval_results)
