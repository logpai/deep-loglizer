import os
import io
import itertools
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from deeploglizer.common.utils import (
    json_pretty_dump,
    dump_pickle,
    load_pickle,
)
import hashlib
import pickle
import json
import re

from IPython import embed


def load_vectors(fname):
    if fname.endswith("pkl"):
        with open(fname, "rb") as fr:
            data = pickle.load(fr)
    else:
        # load fasttext file
        fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(" ")
            data[tokens[0]] = map(float, tokens[1:])
    print("Loading vectors from {} done.".format(fname))

    # with open("../data/pretrain/wiki-news-300d-1M.pkl", "wb") as fw:
    #     pickle.dump(data, fw)
    return data


class Vocab:
    def __init__(self, max_token_len, min_token_count, use_tfidf=False):
        self.max_token_len = max_token_len
        self.min_token_count = min_token_count
        self.use_tfidf = use_tfidf
        self.word2idx = {"padding_token": 0, "oov_token": 1}
        self.token_vocab_size = None

    def __tokenize_log(self, log):
        word_lst_tmp = re.findall(r"[a-zA-Z]+", log)
        word_lst = []
        for word in word_lst_tmp:
            res = list(filter(None, re.split("([A-Z][a-z][^A-Z]*)", word)))
            if len(res) == 0:
                word_lst.append(word)
            else:
                res = [w.lower() for w in res]
                word_lst.extend(res)
        return word_lst

    def gen_pretrain_matrix(self, pretrain_path):
        print("Generating a pretrain matrix.")
        word_vec_dict = load_vectors(pretrain_path)
        vocab_size = len(self.word2idx)
        pretrain_matrix = np.zeros([vocab_size, 300])
        oov_count = 0
        for word, idx in self.word2idx.items():
            if word in word_vec_dict:
                pretrain_matrix[idx] = word_vec_dict[word]
            else:
                oov_count += 1
        print(
            "{}/{} words are assgined pretrained vectors.".format(
                vocab_size - oov_count, vocab_size
            )
        )
        return pretrain_matrix

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
        valid_tokens = set(
            [
                word
                for word, count in token_counter.items()
                if count >= self.min_token_count
            ]
        )
        self.word2idx.update({word: idx for idx, word in enumerate(valid_tokens, 2)})
        self.token_vocab_size = len(self.word2idx)

    def fit_tfidf(self, total_logs):
        print("Fitting tfidf.")
        self.tfidf = TfidfVectorizer(
            tokenizer=lambda x: self.__tokenize_log(x),
            vocabulary=self.word2idx,
            norm="l1",
        )
        self.tfidf.fit(total_logs)

    def transform_tfidf(self, logs):
        return self.tfidf.transform(logs)

    def logs2idx(self, logs):
        idx_list = []
        for log in logs:
            tokens = self.__tokenize_log(log)
            tokens_idx = self.trp(
                [self.word2idx.get(t, 1) for t in tokens], self.max_token_len
            )
            idx_list.append(tokens_idx)
        return idx_list


class FeatureExtractor(BaseEstimator):
    """
    feature_type: "sequentials", "semantics", "quantitatives"
    window_type: "session", "sliding"
    max_token_len: only used for semantics features
    """

    def __init__(
        self,
        label_type="next_log",  # "none", "next_log", "anomaly"
        feature_type="sequentials",
        eval_type="session",
        window_type="sliding",
        window_size=None,
        stride=None,
        max_token_len=50,
        min_token_count=1,
        pretrain_path=None,
        use_tfidf=False,
        deduplicate_windows=True,
        cache=False,
    ):
        self.label_type = label_type
        self.feature_type = feature_type
        self.eval_type = eval_type
        self.window_type = window_type
        self.window_size = window_size
        self.stride = stride
        self.pretrain_path = pretrain_path
        self.use_tfidf = use_tfidf
        self.max_token_len = max_token_len
        self.min_token_count = min_token_count
        self.deduplicate_windows = deduplicate_windows
        self.cache = cache
        self.vocab = Vocab(max_token_len, min_token_count)
        self.meta_data = {}

        if cache:
            param_json = self.get_params()
            identifier = hashlib.md5(str(param_json).encode("utf-8")).hexdigest()[0:8]
            self.cache_dir = os.path.join("./cache", identifier)
            os.makedirs(self.cache_dir, exist_ok=True)
            json_pretty_dump(
                param_json, os.path.join(self.cache_dir, "feature_extractor.json")
            )

    def __generate_windows(self, session_dict, stride):
        window_count = 0
        for session_id, data_dict in session_dict.items():
            if self.window_type == "sliding":
                i = 0
                templates = data_dict["templates"]
                template_len = len(templates)
                windows = []
                window_labels = []
                window_anomalies = []
                while i + self.window_size < template_len:
                    window = templates[i : i + self.window_size]
                    next_log = self.log2id_train.get(templates[i + self.window_size], 1)

                    if session_id == "all":
                        window_anomaly = int(
                            1 in data_dict["label"][i : i + self.window_size]
                        )
                    else:
                        window_anomaly = data_dict["label"]

                    windows.append(window)
                    window_labels.append(next_log)
                    window_anomalies.append(window_anomaly)
                    i += stride
                else:
                    window = templates[i:-1]
                    window.extend(["padding_token"] * (self.window_size - len(window)))
                    next_log = self.log2id_train.get(templates[-1], 1)

                    if session_id == "all":
                        window_anomaly = int(1 in data_dict["label"][i:-1])
                    else:
                        window_anomaly = data_dict["label"]

                    windows.append(window)
                    window_labels.append(next_log)
                    window_anomalies.append(window_anomaly)

                if self.deduplicate_windows:
                    print("Deduplicating windows...")
                    windows, uidx = np.unique(windows, axis=0, return_index=True)
                    window_labels = np.array(window_labels)[uidx]
                    window_anomalies = np.array(window_anomalies)[uidx]
                window_count += len(windows)

                session_dict[session_id]["windows"] = windows
                session_dict[session_id]["window_labels"] = window_labels
                session_dict[session_id]["window_anomalies"] = window_anomalies

            elif self.window_type == "session":
                session_dict[session_id]["windows"] = data_dict["templates"]
                session_dict[session_id]["window_labels"] = [data_dict["label"]]
                window_count += 1

        print("{} sliding windows generated.".format(window_count))

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
        total_idx = []
        for window in windows:
            if self.use_tfidf:
                total_idx.append(self.vocab.transform_tfidf(window).toarray())
            else:
                total_idx.append(self.vocab.logs2idx(window))
        return np.array(total_idx)

    def save(self):
        print("Saving feature extractor to {}.".format(self.cache_dir))
        with open(os.path.join(self.cache_dir, "est.pkl"), "wb") as fw:
            pickle.dump(self, fw)

    def load(self):
        try:
            save_file = os.path.join(self.cache_dir, "est.pkl")
            print("Loading feature extractor from {}.".format(save_file))
            with open(save_file, "rb") as fw:
                obj = pickle.load(fw)
                self.__dict__ = obj.__dict__
                return True
        except Exception as e:
            print("Cannot load cached feature extractor.")
            return False

    def fit(self, session_dict):
        if self.load():
            return
        log_padding = "<pad>"
        log_oov = "<oov>"

        # encode
        total_logs = list(
            itertools.chain(*[v["templates"] for k, v in session_dict.items()])
        )
        self.ulog_train = set(total_logs)
        self.id2log_train = {0: log_padding, 1: log_oov}
        self.id2log_train.update(
            {idx: log for idx, log in enumerate(self.ulog_train, 2)}
        )
        self.log2id_train = {v: k for k, v in self.id2log_train.items()}

        print("{} tempaltes are found.".format(len(self.log2id_train)))

        if self.label_type == "next_log":
            self.meta_data["num_labels"] = len(self.log2id_train)
        elif self.label_type == "anomaly":
            self.meta_data["num_labels"] = 2

        if self.feature_type == "semantics":
            print("Using semantics.")
            print("Building vocab.")
            self.vocab.build_vocab(self.ulog_train)
            print("Building vocab done.")
            self.meta_data["vocab_size"] = self.vocab.token_vocab_size

            if self.pretrain_path is not None:
                self.meta_data["pretrain_matrix"] = self.vocab.gen_pretrain_matrix(
                    self.pretrain_path
                )
            if self.use_tfidf:
                self.vocab.fit_tfidf(total_logs)

        elif self.feature_type == "sequentials":
            self.meta_data["vocab_size"] = len(self.log2id_train)

        if self.cache:
            self.save()

    def transform(self, session_dict, datatype="train"):
        print("Transforming {} data.".format(datatype))
        if datatype == "test":
            # handle new logs
            ulog_test = set(
                itertools.chain(*[v["templates"] for k, v in session_dict.items()])
            )
            ulog_new = ulog_test - self.ulog_train
            print(f"{len(ulog_new)} new templates show while testing.")

        if self.cache:
            cached_file = os.path.join(self.cache_dir, datatype + ".pkl")
            if os.path.isfile(cached_file):
                return load_pickle(cached_file)

        # generate windows, each window contains logid only
        if datatype == "train":
            self.__generate_windows(session_dict, self.stride)
        else:
            self.__generate_windows(session_dict, self.stride)

        # for each window
        for session_id, data_dict in session_dict.items():
            feature_dict = defaultdict(list)
            windows = data_dict["windows"]

            # generate sequential feautres # sliding windows on logid list
            if self.feature_type == "sequentials":
                feature_dict["sequentials"] = self.__windows2sequential(windows)

            # generate semantics feautres # use logid -> token id list
            if self.feature_type == "semantics":
                feature_dict["semantics"] = self.__window2semantics(windows)

            # generate quantitative feautres # count logid in each window
            if self.feature_type == "quantitatives":
                feature_dict["quantitatives"] = self.__windows2quantitative(windows)

            session_dict[session_id]["features"] = feature_dict

        print("Finish feature extraction.")
        if self.cache:
            dump_pickle(session_dict, cached_file)
        return session_dict

    def fit_transform(self, session_dict):
        self.fit(session_dict)
        return self.transform(session_dict, datatype="train")
