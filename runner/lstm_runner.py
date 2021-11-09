#########
# Inw DS 2021
# Run an LSTM model on preprocessed data
#  
#########

import argparse
from torch.utils.data import DataLoader
import logging
from pprint import pprint

from deeploglizer.models import LSTM
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.dataloader import load_sessions, LogDataset
from deeploglizer.common.utils import seed_everything, dump_final_results, dump_params

from deeploglizer.model_configs import LSTMConfig, lstm_config_keys, LogDataSetInputObject

logging.basicConfig(level=logging.INFO, format="%(message)s")  # better control with this
logger = logging.getLogger("anomalydetector")
logger.setLevel(level=logging.INFO)


def run_lstm(params: dict) -> dict:
    """run lstm demo according to parameters"""

    model_save_path = dump_params(params)

    seed_everything(params["random_seed"])

    # data ... 

    # load session data from disk
    session_train, session_test = load_sessions(data_dir=params["data_dir"])

    # Extract Features
    ext = FeatureExtractor(**params)
    session_train = ext.fit_transform(session_train)
    session_test = ext.transform(session_test, datatype="test")

    # Validate one of the sessions
    datadict = next(iter(session_test.values()))
    assert LogDataSetInputObject(**datadict)

    # Build dataset to be fed to the model
    dataset_train = LogDataset(session_train, feature_type=params["feature_type"])
    dataloader_train = DataLoader(dataset_train, batch_size=params["batch_size"], shuffle=True, pin_memory=True)

    dataset_test = LogDataset(session_test, feature_type=params["feature_type"])
    # why is shuffle false and batch_size hardcoded ?
    dataloader_test = DataLoader(dataset_test, batch_size=4096, shuffle=False, pin_memory=True)

    # initialize model using Config object to validate parameters ...
    logging.info(
        f"Provided run parameters not processed by LSTMConfig: {[k for k in params.keys() if k not in lstm_config_keys]}"
    )
    pars = {k: v for k, v in params.items() if k in lstm_config_keys}
    pars["model_save_path"] = model_save_path
    pars["meta_data"] = ext.meta_data
    
    lstm_cfg = LSTMConfig(**pars)
    model = LSTM(**vars(lstm_cfg))

    # train model ...

    eval_results = model.fit(
        dataloader_train,
        test_loader=dataloader_test,
        epoches=params["epoches"],
        learning_rate=params["learning_rate"],
    )

    dump_final_results(params, eval_results, model)

    return eval_results


def get_result_formatted(eval_results, params):
    # report formatting ---------------

    result_str = "\t".join(["{}-{:.4f}".format(k, v) for k, v in eval_results.items()])
    key_info = [
        "dataset",
        "train_anomaly_ratio",
        "feature_type",
        "label_type",
        "use_attention",
    ]
    args_str = "\t".join(["{}:{}".format(k, v) for k, v in params.items() if k in key_info])

    return result_str, args_str


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ##### Model params
    parser.add_argument("--model_name", default="LSTM", type=str)
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--num_directions", default=2, type=int)
    parser.add_argument("--embedding_dim", default=32, type=int)

    ##### Dataset params
    parser.add_argument("--dataset", default="HDFS", type=str)
    parser.add_argument("--data_dir", default="../data/processed/HDFS_100k/hdfs_0.0_tar", type=str)
    parser.add_argument("--window_size", default=10, type=int)
    parser.add_argument("--stride", default=1, type=int)

    ##### Input params
    parser.add_argument("--feature_type", default="sequentials", type=str, choices=["sequentials", "semantics"])
    parser.add_argument("--label_type", default="next_log", type=str)
    parser.add_argument("--use_tfidf", action="store_true")
    parser.add_argument("--max_token_len", default=50, type=int)
    parser.add_argument("--min_token_count", default=1, type=int)
    # Uncomment the following to use pretrained word embeddings. The "embedding_dim" should be set as 300
    # parser.add_argument(
    #     "--pretrain_path", default="../data/pretrain/wiki-news-300d-1M.vec", type=str
    # )

    ##### Training params
    parser.add_argument("--epoches", default=100, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--patience", default=3, type=int)

    ##### Others
    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--gpu", default=0, type=int)

    params = vars(parser.parse_args())

    run_lstm(params)
