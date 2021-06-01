# --embedding_dim 8
# --nhead 4
# --hidden_size 100
# --num_layers 1

# wo semantics
python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --embedding_dim 2 --nhead 1 --gpu 0 B


python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --embedding_dim 4 --nhead 1 --gpu 0 B


python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --embedding_dim 8 --nhead 1 --gpu 0 B


python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --embedding_dim 2 --nhead 2 --gpu 1 B


python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --embedding_dim 2 --nhead 4 --gpu 1 B


python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --embedding_dim 2 --nhead 8 --gpu 1 B



