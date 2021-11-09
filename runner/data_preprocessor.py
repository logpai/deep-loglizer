#########
# Inw DS 2021
# Preprocess raw logdata to produce data to run a model
#  
#########
import os
import re
import pickle
import argparse
import pandas as pd
import numpy as np
from collections import OrderedDict, defaultdict
from typing import Tuple

from data_preprocess.preprocess_hdfs import preprocess_hdfs
from data_preprocess.utils import decision, json_pretty_dump


class Preprocessor:
    """ Generic Preprocessor
    """
    def __init__(self, params:dict):
        pass

    def preprocess(self, logs:pd.DataFrame) -> Tuple[dict, dict]:
        """ Process Structured Logs into a tuple of dictionaries [train, test], in deep-loglizer format 
        """

        return

    def save_to_disk(self):
        """ Saves deep loglizer dataset to disk as pickles
        """

        return

    def load_from_disk(self):
        """ Loads deep loglizer dataset from pickles in disk
        """

        return


class HDFSPreprocessor(Preprocessor):
    """ Preprocessor for HDFS, wrapper around deep-loglize
    """
    def __init__(self, params:dict):
        super().__init__()
        self.params = params
        self._setup()
    
    def _setup(self):

        seed = 42
        np.random.seed(seed)

        params = self.params

        data_name = f'hdfs_{params["train_anomaly_ratio"]}_tar'
        data_dir = "../data/processed/HDFS_100k"

        data_dir = os.path.join(data_dir, data_name)
        os.makedirs(data_dir, exist_ok=True)
      
        return

    def preprocess_deeploglize(self):
        """ wrapper around deep-loglize function 
            - loads csv with HDFS parsed data (i.e. with log keys -> templates)
            - assigns labels if found
            - groups by session id
            - partitions between train and test datasets
            - injects anomalies according to ratio in params
            - saves to pickles and auxiliary json

        Returns
        -------

        each returned session_[train/test] object has the following structure:

        Dict[session_id: SessionDict]

        SessionDict: PreProcessedObject with keys:
                    label :int: `1`` if anomaly, else ``0`` 
                    templates: Sequential List of logkey templates in each session.
        """

        sessions_train, sessions_test = preprocess_hdfs(**self.params)

        return 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_anomaly_ratio", default=1.0, type=float)
    params = vars(parser.parse_args())

    params = {
        # "log_file": "../data/HDFS/HDFS.log_structured.csv",
        "log_file": "../data/HDFS/HDFS_100k.log_structured.csv",
        "label_file": "../data/HDFS/anomaly_label.csv",
        "test_ratio": 0.2,
        "random_sessions": True,  # shuffle sessions
        "train_anomaly_ratio": params["train_anomaly_ratio"],
    }


    proc = HDFSPreprocessor(params)
    proc.preprocess_deeploglize()



    
