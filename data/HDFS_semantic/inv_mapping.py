import os
import pandas as pd
import pickle


def convert(filepath, mapping):
    session = {}
    for idx, line in enumerate(open(filepath)):
        sample = {"templates": [mapping[item] for item in line.split()], "label": 0}
        session[idx] = sample
    return session


log_id_path = "../HDFS_id"
train = os.path.join(log_id_path, "hdfs_train")
test_normal = os.path.join(log_id_path, "hdfs_test_normal")
test_anomaly = os.path.join(log_id_path, "hdfs_test_abnormal")
mapping_file = os.path.join("col_header.txt")

mapping = {}
for line in open(mapping_file):
    tokens = line.split(".")
    idx = tokens[0]
    strs = ".".join(tokens[1:])
    mapping[idx] = strs


session_train = convert(train, mapping)
session_test = {}
session_test_normal = convert(test_normal, mapping)
session_test_anomaly = convert(test_anomaly, mapping)
session_test.update(session_test_normal)
session_test.update(session_test_anomaly)

with open("session_train.pkl", "wb") as fw:
    pickle.dump(session_train, fw)

with open("session_test.pkl", "wb") as fw:
    pickle.dump(session_test, fw)

