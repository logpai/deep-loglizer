#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

sys.path.append("../")
import argparse
from deeploglizer.models import LSTM
from deeploglizer.common.dataloader import load_sessions, log_dataset
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.utils import seed_everything, set_device, dump_params
from torch.utils.data import DataLoader

from IPython import embed

# python deeplog_demo.py --test_ratio 0.2 --train_anomaly_ratio 0 --feature_type sequentials --dataset BGL --label_type next_log --gpu 0 > logs/deeplog.0 2>&1 &

parser = argparse.ArgumentParser()

##################### fixed for Deeplog ↓ ####################
parser.add_argument("--feature_type", default="sequentials", type=str)
parser.add_argument("--use_attention", action="store_true")
parser.add_argument("--label_type", default="next_log", type=str)
parser.add_argument("--use_tfidf", action="store_true")
parser.add_argument("--pretrain_path", default=None, type=str)
##################### fixed for Deeplog ↑ ####################

##### for semantics:
parser.add_argument("--max_token_len", default=50, type=int)
parser.add_argument("--min_token_count", default=1, type=int)

##### model params:
parser.add_argument("--hidden_size", default=128, type=int)
parser.add_argument("--num_directions", default=1, type=float)
parser.add_argument("--embedding_dim", default=8, type=int)

##### dataset params
parser.add_argument("--dataset", default="BGL", type=str)
parser.add_argument("--train_anomaly_ratio", default=0, type=float)
parser.add_argument("--random_partition", action="store_true")
parser.add_argument("--train_ratio", default=None, type=float)
parser.add_argument("--test_ratio", default=0.8, type=float)
parser.add_argument("--window_size", default=10, type=int)
parser.add_argument("--stride", default=1, type=int)

##### training params
parser.add_argument("--epoches", default=5, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--topk", default=10, type=int)

##### Others
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--gpu", default=0, type=int)

params = vars(parser.parse_args())

pkl_dir = "../data/processed/HDFS/hdfs_no_train_anomaly_8_2"
# pkl_dir = "../data/processed/BGL/bgl_no_train_anomaly_8_2"

model_save_path, hash_id = dump_params(params)

if __name__ == "__main__":
    seed_everything(params["random_seed"])

    session_train, session_test = load_sessions(pkl_dir=pkl_dir)

    ext = FeatureExtractor(**params)

    session_train = ext.fit_transform(session_train)
    session_test = ext.transform(session_test, datatype="test")

    dataset_train = log_dataset(session_train, feature_type=params["feature_type"])
    dataloader_train = DataLoader(
        dataset_train, batch_size=params["batch_size"], shuffle=True, pin_memory=True
    )

    dataset_test = log_dataset(session_test, feature_type=params["feature_type"])
    dataloader_test = DataLoader(
        dataset_test, batch_size=4096, shuffle=False, pin_memory=True
    )

    model = LSTM(meta_data=ext.meta_data, model_save_path=model_save_path, **params)

    eval_results = model.fit(
        dataloader_train,
        test_loader=dataloader_test,
        epoches=params["epoches"],
        learning_rate=params["learning_rate"],
    )

    result_str = "\t".join(["{}-{:.4f}".format(k, v) for k, v in eval_results.items()])

    key_info = [
        "dataset",
        "train_anomaly_ratio",
        "feature_type",
        "label_type",
        "use_attention",
    ]

    args_str = "\t".join(
        ["{}:{}".format(k, v) for k, v in params.items() if k in key_info]
    )

    with open(os.path.join(f"{params['dataset']}.txt"), "a+") as fw:
        info = "{} DeepLog {} {} train: {:.3f} test: {:.3f}\n".format(
            hash_id,
            args_str,
            result_str,
            model.time_tracker["train"],
            model.time_tracker["test"],
        )
        fw.write(info)