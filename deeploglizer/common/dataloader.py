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
import pickle
import json
from collections import OrderedDict, defaultdict
from typing import Tuple
from torch.utils.data import Dataset
from loguru import logger
from deeploglizer.common.utils import decision


def load_sessions(data_dir:str) -> Tuple[dict, dict]:
    """ load sessions from a data directory

        expects data_desc.json, session_train.pkl and session_test.pkl

        each returned session_[train/test] object has the following structure:

        Dict[session_key: SessionDict]

        SessionDict: PreProcessedObject with keys:
                    label :int: `1`` if anomaly, else ``0`` 
                    templates: Sequential List of logkey templates in each session.

    """
    with open(os.path.join(data_dir, "data_desc.json"), "r") as fr:
        data_desc = json.load(fr)
    with open(os.path.join(data_dir, "session_train.pkl"), "rb") as fr:
        session_train = pickle.load(fr)
    with open(os.path.join(data_dir, "session_test.pkl"), "rb") as fr:
        session_test = pickle.load(fr)

    train_labels = [
        v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0)
        for _, v in session_train.items()
    ]
    test_labels = [
        v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0)
        for _, v in session_test.items()
    ]

    num_train = len(session_train)
    ratio_train = sum(train_labels) / num_train
    num_test = len(session_test)
    ratio_test = sum(test_labels) / num_test
    logger.info("Load from {}".format(data_dir))
    logger.info(json.dumps(data_desc, indent=4))
    logger.info(
        "# train sessions {} ({:.2f} anomalies)".format(num_train, ratio_train)
    )
    logger.info("# test sessions {} ({:.2f} anomalies)".format(num_test, ratio_test))

    return session_train, session_test


class LogDataset(Dataset):
    """ Builds pytorch datasets by only extracting needed features
    """
    def __init__(self, session_dict:dict[str, dict], feature_type:str="semantics"):
        """Buil pytorch datasets by only extracting needed features

        Args:
            session_dict: dict with sessions as keys. Each individual session dict expects as keys:
                ['label', 'templates', 'windows', 'window_labels', 'window_anomalies', 'features']
            feature_type:  Defaults to "semantics".
        """
        flatten_data_list = []
        mappings = []
        # flatten all sessions

        for session_idx, (key, data_dict) in enumerate(session_dict.items()):
            features = data_dict["features"][feature_type]
            window_labels = data_dict["window_labels"]
            window_anomalies = data_dict["window_anomalies"]
            for window_idx in range(len(window_labels)):
                sample = {
                    "session_idx": session_idx,  # internal index mapping to session key
                    "features": features[window_idx],
                    "window_labels": window_labels[window_idx],
                    "window_anomalies": window_anomalies[window_idx],
                }
                flatten_data_list.append(sample)
            
            mappings.append((session_idx, key))
    
        self.flatten_data_list = flatten_data_list
        mappings = list(set(mappings))
        self.mapping = { k: v for k, v in  zip([m[0] for m in mappings], [m[1] for m in mappings])}

    def __len__(self):
        return len(self.flatten_data_list)

    def __getitem__(self, idx):
        return self.flatten_data_list[idx]


def load_BGL(
    log_file,
    train_ratio=None,
    test_ratio=0.8,
    train_anomaly_ratio=0,
    random_partition=False,
    filter_normal=True,
    **kwargs
):
    logger.info("Loading BGL logs from {}.".format(log_file))
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    # struct_log.sort_values(by=["Timestamp"], inplace=True)
    logger.info("{} lines loaded.".format(struct_log.shape[0]))

    templates = struct_log["EventTemplate"].values
    labels = struct_log["Label"].map(lambda x: x != "-").astype(int).values

    total_indice = np.array(list(range(templates.shape[0])))
    if random_partition:
        logger.info("Using random partition.")
        np.random.shuffle(total_indice)

    if train_ratio is None:
        train_ratio = 1 - test_ratio
    assert train_ratio + test_ratio <= 1, "train_ratio + test_ratio should <= 1."
    train_lines = int(train_ratio * len(total_indice))
    test_lines = int(test_ratio * len(total_indice))

    idx_train = total_indice[0:train_lines]
    idx_test = total_indice[-test_lines:]

    idx_train = [
        idx
        for idx in idx_train
        if (labels[idx] == 0 or (labels[idx] == 1 and decision(train_anomaly_ratio)))
    ]

    if filter_normal:
        logger.info(
            "Filtering unseen normal tempalates in {} test data.".format(len(idx_test))
        )
        seen_normal = set(templates[idx_train].tolist())
        idx_test = [
            idx
            for idx in idx_test
            if not (labels[idx] == 0 and (templates[idx] not in seen_normal))
        ]

    session_train = {
        "all": {"templates": templates[idx_train].tolist(), "label": labels[idx_train]}
    }
    session_test = {
        "all": {"templates": templates[idx_test].tolist(), "label": labels[idx_test]}
    }

    labels_train = labels[idx_train]
    labels_test = labels[idx_test]

    train_anomaly = 100 * sum(labels_train) / len(labels_train)
    test_anomaly = 100 * sum(labels_test) / len(labels_test)

    logger.info("# train lines: {} ({:.2f}%)".format(len(labels_train), train_anomaly))
    logger.info("# test lines: {} ({:.2f}%)".format(len(labels_test), test_anomaly))

    return session_train, session_test


def load_HDFS(
    log_file,
    label_file,
    train_ratio=None,
    test_ratio=None,
    train_anomaly_ratio=1,
    random_partition=False,
    **kwargs
):
    """Load HDFS structured log into train and test data

    Arguments
    ---------
        TODO

    Returns
    -------
        TODO
    """
    logger.info("Loading HDFS logs from {}.".format(log_file))
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    # struct_log.sort_values(by=["Date", "Time"], inplace=True)

    # assign labels
    label_data = pd.read_csv(label_file, engine="c", na_filter=False, memory_map=True)
    label_data["Label"] = label_data["Label"].map(lambda x: int(x == "Anomaly"))
    label_data_dict = dict(zip(label_data["BlockId"], label_data["Label"]))

    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
    for _, row in enumerate(struct_log.values):
        blkId_list = re.findall(r"(blk_-?\d+)", row[column_idx["Content"]])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if blk_Id not in session_dict:
                session_dict[blk_Id] = defaultdict(list)
            session_dict[blk_Id]["templates"].append(row[column_idx["EventTemplate"]])

    for k in session_dict.keys():
        session_dict[k]["label"] = label_data_dict[k]

    session_idx = list(range(len(session_dict)))
    # split data
    if random_partition:
        logger.info("Using random partition.")
        np.random.shuffle(session_idx)

    session_ids = np.array(list(session_dict.keys()))
    session_labels = np.array(list(map(lambda x: label_data_dict[x], session_ids)))

    if train_ratio is None:
        train_ratio = 1 - test_ratio
    train_lines = int(train_ratio * len(session_idx))
    test_lines = int(test_ratio * len(session_idx))

    session_idx_train = session_idx[0:train_lines]
    session_idx_test = session_idx[-test_lines:]

    session_id_train = session_ids[session_idx_train]
    session_id_test = session_ids[session_idx_test]
    session_labels_train = session_labels[session_idx_train]
    session_labels_test = session_labels[session_idx_test]

    logger.info("Total # sessions: {}".format(len(session_ids)))

    session_train = {
        k: session_dict[k]
        for k in session_id_train
        if (session_dict[k]["label"] == 0)
        or (session_dict[k]["label"] == 1 and decision(train_anomaly_ratio))
    }

    session_test = {k: session_dict[k] for k in session_id_test}

    session_labels_train = [v["label"] for k, v in session_train.items()]
    session_labels_test = [v["label"] for k, v in session_test.items()]

    train_anomaly = 100 * sum(session_labels_train) / len(session_labels_train)
    test_anomaly = 100 * sum(session_labels_test) / len(session_labels_test)

    logger.info(
        "# train sessions: {} ({:.2f}%)".format(len(session_train), train_anomaly)
    )
    logger.info(
        "# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly)
    )

    return session_train, session_test


def load_HDFS_semantic(log_semantic_path):
    train = os.path.join(log_semantic_path, "session_train.pkl")
    test = os.path.join(log_semantic_path, "session_test.pkl")

    with open(train, "rb") as fr:
        session_train = pickle.load(fr)

    with open(test, "rb") as fr:
        session_test = pickle.load(fr)

    # session_test = {k: v for i, (k, v) in enumerate(session_test.items()) if i < 50000}
    logger.info(
        "# train sessions: {}, # test sessions: {}".format(
            len(session_train), len(session_test)
        )
    )
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

    logger.info(
        "# train sessions: {}, # test sessions: {}".format(
            len(session_train), len(session_test)
        )
    )

    # logger.info("# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly_ratio))
    return session_train, session_test
