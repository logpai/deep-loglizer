#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../")
from deeploglizer.models import LSTM
from deeploglizer.common.dataloader import load_HDFS, log_dataset
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.utils import seed_everything, set_device
from torch.utils.data import DataLoader

from IPython import embed

random_seed = 42
device = 0

feature_type = "sequentials"  # "sequentials", "semantics", "quantitatives"
sequential_partition = False
test_ratio = 0.2
window_size = 15
stride = 15

topk = 20
batch_size = 512
epoches = 1
learning_rate = 1.0e-3
use_tfidf = False

hidden_size = 32
num_directions = 1
embedding_dim = 5

max_token_len = 50  # max #token for each event [semantic only]
min_token_count = 1  # min # occurrence of token for each event [semantic only]
pretrain_path = None
# pretrain_path = "../data/pretrain/wiki-news-300d-1M.vec"

log_file = "../data/HDFS/HDFS.log_structured.csv"  # The structured log file
# log_file = "../data/HDFS/HDFS_100k.log_structured.csv"  # The structured log file
label_file = "../data/HDFS/anomaly_label.csv"  # The anomaly label file

if __name__ == "__main__":
    seed_everything(random_seed)

    session_train, session_test = load_HDFS(
        log_file,
        label_file=label_file,
        test_ratio=test_ratio,
        sequential_partition=sequential_partition,
        random_seed=random_seed,
    )

    ext = FeatureExtractor(
        label_type="next_log",  # "none", "next_log", "anomaly"
        feature_type=feature_type,  # "sequentials", "semantics", "quantitatives"
        window_type="sliding",
        window_size=window_size,
        stride=stride,
        max_token_len=max_token_len,
        min_token_count=min_token_count,
        pretrain_path=pretrain_path,
        use_tfidf=use_tfidf,
    )

    ext.fit_transform(session_train)
    ext.transform(session_test, datatype="test")

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
        use_tfidf=use_tfidf,
        topk=topk,
        device=device,
    )
    model.fit(dataloader_train, epoches=epoches, learning_rate=learning_rate)

    print("Evaluating train:")
    eval_results = model.evaluate(dataloader_train)
    print(eval_results)

    print("Evaluating test:")
    eval_results = model.evaluate(dataloader_test)
    print(eval_results)
