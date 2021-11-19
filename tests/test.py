from demo.lstm_demo import main as run_lstm

def test_unsup_lstm_on_hdfs():

    params = {

        # model parameters
        "model_name": "LSTM",
        "use_attention": False,
        "hidden_size": 128,
        "num_layers": 2,
        "num_directions": 1, # they had 2
        "embedding_dim": 32,
        "feature_type": "sequentials",
        "label_type": "next_log", # to test the unsup model
        "use_tfidf": False,
        "topk": 10,
        "freeze": False,

        # data parameters
        "dataset": "HDFS",
        "data_dir": "data/processed/HDFS_100k/hdfs_1.0_tar",

        # data preprocessing
        "window_size": 10,
        "stride": 1,
        "max_token_len": 50,
        "min_token_count": 1,

        # training parameters
        "epochs": 100, 
        "batch_size": 1024,
        "learning_rate": 0.01, # using Adam
        "patience": 3, # for early stop

        # running parameters
        "random_seed": 42,
        "gpu": 0,
    }

    out = run_lstm(params)

    print(out)

    assert out
    assert out["f1"] > 0.1
    assert out["f1"] < 0.21

if __name__ == "__main__":

    test_unsup_lstm_on_hdfs()
