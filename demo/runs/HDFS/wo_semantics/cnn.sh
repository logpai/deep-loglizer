# --hidden_size 200
# --num_directions 1
# --embedding_dim 8
# --hidden_size 200
# --kernel_sizes 2 3 4
# --embedding_dim 8

# wo semantics
python cnn_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10

