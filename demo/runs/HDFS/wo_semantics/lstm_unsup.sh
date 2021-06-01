# --hidden_size 200
# --num_directions 1
# --embedding_dim 8
# --num_layers 1

# wo semantics
python lstm_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --label_type next_log --window_size 10 --topk 10
