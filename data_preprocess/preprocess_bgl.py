from logging import currentframe
import os
import re
import pickle
import pandas as pd
import numpy as np
from utils import decision, json_pretty_dump
from collections import OrderedDict, defaultdict


eval_name = "bgl_no_train_anomaly_8_2"
seed = 42
outdir = "../data/processed/BGL"
np.random.seed(seed)

params = {
    # "log_file": "../data/BGL/BGL_100k.log_structured.csv",
    "log_file": "../data/BGL/BGL.log_groundtruth.csv",
    "time_range": 60,  # 60 seconds
    "train_ratio": None,
    "test_ratio": 0.2,
    "random_sessions": True,
    "train_anomaly_ratio": 0,
}

outdir = os.path.join(outdir, eval_name)
os.makedirs(outdir, exist_ok=True)


def load_BGL(
    log_file,
    time_range,
    train_ratio,
    test_ratio,
    random_sessions,
    train_anomaly_ratio,
):
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    # struct_log.sort_values(by=["Timestamp"], inplace=True)
    print("{} lines loaded.".format(struct_log.shape[0]))

    struct_log["Label"] = struct_log["Label"].map(lambda x: x != "-").astype(int).values
    struct_log["time"] = pd.to_datetime(
        struct_log["Time"], format="%Y-%m-%d-%H.%M.%S.%f"
    )
    struct_log["seconds_since"] = (
        (struct_log["time"] - struct_log["time"][0]).dt.total_seconds().astype(int)
    )

    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
    for idx, row in enumerate(struct_log.values):
        current = row[column_idx["seconds_since"]]
        if idx == 0:
            sessid = current
        elif current - sessid > time_range:
            sessid = current
        if sessid not in session_dict:
            session_dict[sessid] = defaultdict(list)
        session_dict[sessid]["templates"].append(row[column_idx["EventTemplate"]])
        session_dict[sessid]["labels"].append(row[column_idx["Label"]])

    for k, v in session_dict.items():
        session_dict[k]["label"] = int(1 in v["labels"])
        del session_dict[k]["labels"]

    session_idx = list(range(len(session_dict)))
    # split data
    if random_sessions:
        print("Using random partition.")
        np.random.shuffle(session_idx)

    session_ids = np.array(list(session_dict.keys()))

    if train_ratio is None:
        train_ratio = 1 - test_ratio
    train_lines = int(train_ratio * len(session_idx))
    test_lines = int(test_ratio * len(session_idx))

    session_idx_train = session_idx[0:train_lines]
    session_idx_test = session_idx[-test_lines:]

    session_id_train = session_ids[session_idx_train]
    session_id_test = session_ids[session_idx_test]

    print("Total # sessions: {}".format(len(session_ids)))

    session_train = {
        k: session_dict[k]
        for k in session_id_train
        if (session_dict[k]["label"] == 0)
        or (session_dict[k]["label"] == 1 and decision(train_anomaly_ratio))
    }
    session_test = {k: session_dict[k] for k in session_id_test}

    session_labels_train = [v["label"] for k, v in session_train.items()]
    session_labels_test = [v["label"] for k, v in session_test.items()]

    print(len(session_labels_train))
    print(len(session_labels_test))

    train_anomaly = 100 * sum(session_labels_train) / len(session_labels_train)
    test_anomaly = 100 * sum(session_labels_test) / len(session_labels_test)

    print("# train sessions: {} ({:.2f}%)".format(len(session_train), train_anomaly))
    print("# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly))

    with open(os.path.join(outdir, "session_train.pkl"), "wb") as fw:
        pickle.dump(session_train, fw)
    with open(os.path.join(outdir, "session_test.pkl"), "wb") as fw:
        pickle.dump(session_test, fw)
    json_pretty_dump(params, os.path.join(outdir, "data_desc.json"))
    return session_train, session_test


if __name__ == "__main__":
    load_BGL(**params)