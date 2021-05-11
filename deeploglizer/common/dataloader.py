"""
The interface to load log datasets. The datasets currently supported include
HDFS and BGL.

Authors:
    LogPAI Team

"""

import pandas as pd
import os
import numpy as np
import re
from sklearn.utils import shuffle
from collections import OrderedDict, defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class log_dataset(Dataset):
    def __init__(self, session_dict, feature_type="semantics"):
        flatten_data_list = []
        # flatten all sessions
        for session_idx, data_dict in enumerate(session_dict.values()):
            features = data_dict["features"][feature_type]
            labels = data_dict["window_labels"]
            for window_idx in range(len(labels)):
                sample = {
                    "session_idx": session_idx,  # not session id
                    "features": features[window_idx],
                    "window_labels": labels[window_idx],
                    "session_labels": data_dict["label"],
                }
                flatten_data_list.append(sample)
        self.flatten_data_list = flatten_data_list
        print("Finish data preprocessing.")

    def __len__(self):
        return len(self.flatten_data_list)

    def __getitem__(self, idx):
        return self.flatten_data_list[idx]


def load_HDFS(
    log_file,
    label_file,
    test_ratio=None,
    first_n_rows=100000,
    sequential_partition=True,
    random_seed=42,
):
    """Load HDFS structured log into train and test data

    Arguments
    ---------
        TODO

    Returns
    -------
        TODO
    """
    print("Loading logs from {}.".format(log_file))
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    # struct_log.sort_values(by=["Date", "Time"], inplace=True)

    # assign labels
    label_data = pd.read_csv(label_file, engine="c", na_filter=False, memory_map=True)
    label_data["Label"] = label_data["Label"].map(lambda x: int(x == "Anomaly"))
    label_data_dict = dict(zip(label_data["BlockId"], label_data["Label"]))

    if first_n_rows is None:
        session_dict = OrderedDict()
        column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
        for idx, row in enumerate(struct_log.values):
            blkId_list = re.findall(r"(blk_-?\d+)", row[column_idx["Content"]])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if blk_Id not in session_dict:
                    session_dict[blk_Id] = defaultdict(list)
                session_dict[blk_Id]["templates"].append(
                    row[column_idx["EventTemplate"]]
                )

        for k in session_dict.keys():
            session_dict[k]["label"] = label_data_dict[k]
        # split data
        session_ids = list(session_dict.keys())
        session_labels = list(map(lambda x: label_data_dict[x], session_ids))
        (
            session_id_train,
            session_id_test,
            session_labels_train,
            session_labels_test,
        ) = train_test_split(
            session_ids,
            session_labels,
            test_size=test_ratio,
            shuffle=(sequential_partition == False),
            random_state=random_seed,
        )

        session_train = {
            k: session_dict[k]
            for k in session_id_train
            if session_dict[k]["label"] == 0
        }
        session_test = {k: session_dict[k] for k in session_id_test}
    else:
        print("Using first {} rows to build training data.".format(first_n_rows))
        session_train = OrderedDict()
        session_test = OrderedDict()
        column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
        blk_count = 0
        for idx, row in enumerate(struct_log.values):
            blkId_list = re.findall(r"(blk_-?\d+)", row[column_idx["Content"]])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if idx < first_n_rows:
                    if blk_Id not in session_train:
                        session_train[blk_Id] = defaultdict(list)
                    session_train[blk_Id]["templates"].append(
                        row[column_idx["EventTemplate"]]
                    )
                else:
                    if blk_Id not in session_test:
                        session_test[blk_Id] = defaultdict(list)
                        blk_count += 1
                    session_test[blk_Id]["templates"].append(
                        row[column_idx["EventTemplate"]]
                    )
            if blk_count >= 30000:
                break
        session_labels_train = []
        session_labels_test = []

        tmp_dict = defaultdict(list)
        for k in session_train.keys():
            session_train[k]["label"] = label_data_dict[k]
            session_labels_train.append(label_data_dict[k])

        session_train = {k: v for k, v in session_train.items() if v["label"] == 0}

        for k in session_test.keys():
            session_test[k]["label"] = label_data_dict[k]
            session_labels_test.append(label_data_dict[k])

    train_anomaly_ratio = 100 * sum(session_labels_train) / len(session_labels_train)
    test_anomaly_ratio = 100 * sum(session_labels_test) / len(session_labels_test)

    print(
        "# train sessions: {} ({:.2f}%)".format(len(session_train), train_anomaly_ratio)
    )
    print("# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly_ratio))
    return session_train, session_test


def load_HDFS_id(log_id_path):
    train = os.path.join(log_id_path, "hdfs_train")
    test_normal = os.path.join(log_id_path, "hdfs_test_normal")
    test_anomaly = os.path.join(log_id_path, "hdfs_test_abnormal")

    session_train = {}
    for idx, line in enumerate(open(train)):
        sample = {"templates": line.split(), "label": 0}
        session_train[idx] = sample

    session_test = {}
    for idx, line in enumerate(open(test_normal)):
        if idx > 50000:
            break
        sample = {"templates": line.split(), "label": 0}
        session_test[idx] = sample

    for idx, line in enumerate(open(test_anomaly), len(session_test)):
        if idx > 100000:
            break
        sample = {"templates": line.split(), "label": 1}
        session_test[idx] = sample

    print(
        "# train sessions: {}, # test sessions: {}".format(
            len(session_train), len(session_test)
        )
    )
    # print("# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly_ratio))
    return session_train, session_test