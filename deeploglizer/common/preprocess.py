import itertools
import numpy as np
from collections import Counter, defaultdict


class Vocab:
    def __init__(self, max_token_len, min_token_count):
        self.max_token_len = max_token_len
        self.min_token_count = min_token_count
        self.word2idx = {"padding_token": 0, "oov_token": 1}

    def __tokenize_log(self, log):
        return log.split()

    def trp(self, l, n):
        """ Truncate or pad a list """
        r = l[:n]
        if len(r) < n:
            r.extend(list([1]) * (n - len(r)))
        return r

    def build_vocab(self, logs):
        token_counter = Counter()
        for log in logs:
            tokens = self.__tokenize_log(log)
            token_counter.update(tokens)
        self.word2idx = {
            word: idx
            for idx, word in enumerate(token_counter, 2)
            if token_counter[word] >= self.min_token_count
        }

    def logs2idx(self, logs):
        idx_list = []
        for log in logs:
            tokens = self.__tokenize_log(log)
            tokens_idx = self.trp(
                [self.word2idx.get(t, 1) for t in tokens], self.max_token_len
            )
            idx_list.append(tokens_idx)
        return idx_list


class FeatureExtractor:
    """
    feature_types: "sequentials", "semantics", "quantitatives"
    window_types: "session", "sliding"
    max_token_len: only used for semantics features
    """

    def __init__(
        self,
        label_types="next_log",  # "none", "next_log", "anomaly"
        feature_types=[
            "sequentials",
        ],  # "sequentials", "semantics", "quantitatives"
        window_types="sliding",
        window_size=None,
        stride=None,
        max_token_len=50,
        min_token_count=1,
    ):
        self.label_types = label_types
        self.feature_types = feature_types
        self.window_types = window_types
        self.window_size = window_size
        self.stride = stride
        self.vocab = Vocab(max_token_len, min_token_count)
        self.meta_data = {}

    def __generate_windows(self, session_dict):
        for session_id, data_dict in session_dict.items():
            if self.window_types == "sliding":
                i = 0
                templates = data_dict["templates"]
                template_len = len(templates)
                windows = []
                window_labels = []
                while i + self.window_size < template_len:
                    windows.append(templates[i : i + self.window_size])
                    if self.label_types == "next_log":
                        window_labels.append(
                            self.log2id_train.get(templates[i + self.window_size], 1)
                        )
                    elif self.label_types == "anomaly":
                        window_labels.append(data_dict["label"])
                    elif self.label_types == "none":
                        window_labels.append(None)
                    i += self.stride
                session_dict[session_id]["windows"] = windows
                session_dict[session_id]["window_labels"] = window_labels
            elif self.window_types == "session":
                session_dict[session_id]["windows"] = data_dict["templates"]
                session_dict[session_id]["window_labels"] = [data_dict["label"]]

    def __windows2quantitative(self, windows):
        total_features = []
        for window in windows:
            feature = [0] * len(self.id2log_train)
            window = [self.log2id_train.get(x, 1) for x in window]
            log_counter = Counter(window)
            for logid, log_count in log_counter.items():
                feature[int(logid)] = log_count
            total_features.append(feature[1:])  # discard the position of padding
        return np.array(total_features)

    def __windows2sequential(self, windows):
        total_features = []
        for window in windows:
            ids = [self.log2id_train.get(x, 1) for x in window]
            total_features.append(ids)
        return np.array(total_features)

    def __window2semantics(self, windows):
        # input: raw windows
        # output: encoded token matrix,
        total_features = []
        for window in windows:
            total_features.append(self.vocab.logs2idx(window))
        return np.array(total_features)

    def fit(self, session_dict):
        log_padding = "<pad>"
        log_oov = "<oov>"

        # encode
        self.ulog_train = set(
            itertools.chain(*[v["templates"] for k, v in session_dict.items()])
        )
        self.id2log_train = {0: log_padding, 1: log_oov}
        self.id2log_train.update(
            {idx: log for idx, log in enumerate(self.ulog_train, 2)}
        )
        self.log2id_train = {v: k for k, v in self.id2log_train.items()}

        if self.label_types == "next_log":
            self.meta_data["num_labels"] = len(self.log2id_train)
        elif self.label_types == "anomaly":
            self.meta_data["num_labels"] = 2

        if "semantics" in self.feature_types:
            print("Building vocab.")
            self.vocab.build_vocab(self.ulog_train)
            print("Building vocab done.")

    def transform(self, session_dict, datatype="train"):
        if datatype == "test":
            # handle new logs
            ulog_test = set(
                itertools.chain(*[v["templates"] for k, v in session_dict.items()])
            )
            ulog_new = ulog_test - self.ulog_train
            print(f"{len(ulog_new)} new templates show while testing.")

        # generate windows, each window contains logid only
        self.__generate_windows(session_dict)

        # for each window
        for session_id, data_dict in session_dict.items():
            feature_dict = defaultdict(list)
            windows = data_dict["windows"]

            # generate sequential feautres # sliding windows on logid list
            feature_dict["sequential"] = self.__windows2sequential(windows)

            # generate semantics feautres # use logid -> token id list
            feature_dict["semantics"] = self.__window2semantics(windows)

            # generate quantitative feautres # count logid in each window
            feature_dict["quantitative"] = self.__windows2quantitative(windows)

            session_dict[session_id]["features"] = feature_dict

    def fit_transform(self, session_dict):
        self.fit(session_dict)
        return self.transform(session_dict, datatype="train")