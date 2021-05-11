#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse

sys.path.append("../")
from deeploglizer.models import LSTM
from deeploglizer.common.dataloader import (
    load_BGL,
    log_dataset,
)
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.utils import seed_everything, set_device
from torch.utils.data import DataLoader

from IPython import embed

random_seed = 42

parser = argparse.ArgumentParser()
parser.add_argument("--test_ratio", default=0.2, type=float, help="test_ratio")
parser.add_argument(
    "--train_anomaly_ratio", default=1, type=float, help="train_anomaly_ratio"
)
parser.add_argument("--gpu", default=0, type=int, help="gpu id")
args = vars(parser.parse_args())

test_ratio = args["test_ratio"]
device = args["gpu"]
train_anomaly_ratio = args["train_anomaly_ratio"]

label_type = "next_log"
eval_type = "window"
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

log_file = "../data/BGL/BGL.log_groundtruth.csv"  # The structured log file
if not os.path.isfile(log_file):
    log_file = "../data/BGL/BGL_100k.log_structured.csv"  # The structured log file

if __name__ == "__main__":
    seed_everything(random_seed)

    session_train, session_test = load_BGL(
        log_file=log_file,
        test_ratio=test_ratio,
        train_anomaly_ratio=train_anomaly_ratio,
        sequential_partition=True,
        random_seed=42,
    )

    ext = FeatureExtractor(
        label_type=label_type,  # "none", "next_log", "anomaly"
        feature_type=feature_type,  # "sequentials", "semantics", "quantitatives"
        eval_type=eval_type,
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
        eval_type=eval_type,
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
