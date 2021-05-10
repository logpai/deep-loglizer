import io
import itertools
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle


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
        return log.split()

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


class FeatureExtractor:
    """
    feature_type: "sequentials", "semantics", "quantitatives"
    window_type: "session", "sliding"
    max_token_len: only used for semantics features
    """

    def __init__(
        self,
        label_types="next_log",  # "none", "next_log", "anomaly"
        feature_type="sequentials",
        window_type="sliding",
        window_size=None,
        stride=None,
        max_token_len=50,
        min_token_count=1,
        pretrain_path=None,
        use_tfidf=False,
    ):
        self.label_types = label_types
        self.feature_type = feature_type
        self.window_type = window_type
        self.window_size = window_size
        self.stride = stride
        self.pretrain_path = pretrain_path
        self.use_tfidf = use_tfidf
        self.vocab = Vocab(max_token_len, min_token_count)
        self.meta_data = {}

    def __generate_windows(self, session_dict, window_size):
        for session_id, data_dict in session_dict.items():
            if self.window_type == "sliding":
                i = 0
                templates = data_dict["templates"]
                template_len = len(templates)
                windows = []
                window_labels = []
                while i + window_size < template_len:
                    windows.append(templates[i : i + window_size])
                    if self.label_types == "next_log":
                        window_labels.append(
                            self.log2id_train.get(templates[i + window_size], 1)
                        )
                    elif self.label_types == "anomaly":
                        window_labels.append(data_dict["label"])
                    elif self.label_types == "none":
                        window_labels.append(None)
                    i += self.stride
                session_dict[session_id]["windows"] = windows
                session_dict[session_id]["window_labels"] = window_labels
            elif self.window_type == "session":
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
        total_idx = []
        for window in windows:
            if self.use_tfidf:
                total_idx.append(self.vocab.transform_tfidf(window).toarray())
            else:
                total_idx.append(self.vocab.logs2idx(window))
        return np.array(total_idx)

    def fit(self, session_dict):
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

        if self.label_types == "next_log":
            self.meta_data["num_labels"] = len(self.log2id_train)
        elif self.label_types == "anomaly":
            self.meta_data["num_labels"] = 2

        if self.feature_type == "semantics":
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

    def transform(self, session_dict, datatype="train"):
        if datatype == "test":
            # handle new logs
            ulog_test = set(
                itertools.chain(*[v["templates"] for k, v in session_dict.items()])
            )
            ulog_new = ulog_test - self.ulog_train
            print(f"{len(ulog_new)} new templates show while testing.")

        # generate windows, each window contains logid only
        if datatype == "train":
            self.__generate_windows(session_dict, self.window_size)
        else:
            self.__generate_windows(session_dict, 1)

        # for each window
        for session_id, data_dict in session_dict.items():
            feature_dict = defaultdict(list)
            windows = data_dict["windows"]

            # generate sequential feautres # sliding windows on logid list
            feature_dict["sequentials"] = self.__windows2sequential(windows)

            # generate semantics feautres # use logid -> token id list
            feature_dict["semantics"] = self.__window2semantics(windows)

            # generate quantitative feautres # count logid in each window
            feature_dict["quantitatives"] = self.__windows2quantitative(windows)

            session_dict[session_id]["features"] = feature_dict

    def fit_transform(self, session_dict):
        self.fit(session_dict)
        return self.transform(session_dict, datatype="train")