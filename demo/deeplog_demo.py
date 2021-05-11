#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

sys.path.append("../")
import argparse
from deeploglizer.models import LSTM
from deeploglizer.common.dataloader import (
    load_HDFS,
    load_BGL,
    log_dataset,
)
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.utils import seed_everything, set_device
from torch.utils.data import DataLoader

from IPython import embed


parser = argparse.ArgumentParser()
parser.add_argument("--test_ratio", default=0.2, type=float, help="test_ratio")
parser.add_argument(
    "--train_anomaly_ratio", default=0, type=float, help="train_anomaly_ratio"
)
parser.add_argument(
    "--feature_type", default="sequentials", type=str, help="feature_type"
)
parser.add_argument("--dataset", default="BGL", type=str, help="dataset")
parser.add_argument("--gpu", default=0, type=int, help="gpu id")
args = vars(parser.parse_args())


test_ratio = args["test_ratio"]
train_anomaly_ratio = args["train_anomaly_ratio"]
device = args["gpu"]
feature_type = args["feature_type"]  # "sequentials", "semantics", "quantitatives"
dataset = args["dataset"]


random_seed = 42
label_type = "next_log"
eval_type = "window" if dataset == "BGL" else "session"
window_size = 10
stride = 1

topk = 8
batch_size = 1024
epoches = 5
learning_rate = 1.0e-2
use_tfidf = False
sequential_partition = False

hidden_size = 200
num_directions = 1
embedding_dim = 8

max_token_len = 50  # max #token for each event [semantic only]
min_token_count = 1  # min # occurrence of token for each event [semantic only]
pretrain_path = None
# pretrain_path = "../data/pretrain/wiki-news-300d-1M.vec"

deduplicate_windows = False
cache = False

if dataset == "HDFS":
    log_file = "../data/HDFS/HDFS.log_groundtruth.csv"
    if not os.path.isfile(log_file):
        log_file = "../data/HDFS/HDFS_100k.log_structured.csv"
    label_file = "../data/HDFS/anomaly_label.csv"
elif dataset == "BGL":
    log_file = "../data/BGL/BGL.log_groundtruth.csv"
    if not os.path.isfile(log_file):
        log_file = "../data/BGL/BGL_100k.log_structured.csv"

if __name__ == "__main__":
    seed_everything(random_seed)

    if dataset == "HDFS":
        session_train, session_test = load_HDFS(
            log_file=log_file,
            label_file=label_file,
            test_ratio=test_ratio,
            train_anomaly_ratio=train_anomaly_ratio,
            sequential_partition=sequential_partition,
            random_seed=random_seed,
        )
    elif dataset == "BGL":
        session_train, session_test = load_BGL(
            log_file=log_file,
            test_ratio=test_ratio,
            train_anomaly_ratio=train_anomaly_ratio,
            sequential_partition=True,
            random_seed=random_seed,
        )

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
        eval_type=eval_type,
        use_tfidf=use_tfidf,
        topk=topk,
        device=device,
    )

    eval_results = model.fit(
        dataloader_train,
        test_loader=dataloader_test,
        epoches=epoches,
        learning_rate=learning_rate,
    )

    result_str = "\t".join(["{}-{:.4f}".format(k, v) for k, v in eval_results.items()])
    args_str = "\t".join(["{}:{}".format(k, v) for k, v in args.items()])
    os.makedirs("./demo_results/", exist_ok=True)
    with open(os.path.join("./demo_results/", f"HDFS_deeplog.txt"), "a+") as fw:
        info = "{} {}\n".format(args_str, result_str)
        fw.write(info)