# --embedding_dim 8
# --nhead 4
# --hidden_size 100
# --num_layers 1

# wo semantics
python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --feature_type semantics --use_tfidf B

python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --feature_type semantics --use_tfidf --embedding_dim 4 --nhead 1  --hidden_size 100 --gpu 0 B

python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --feature_type semantics --use_tfidf --embedding_dim 8 --nhead 1  --hidden_size 100 --gpu 0 B

python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --feature_type semantics --use_tfidf --embedding_dim 16 --nhead 1  --hidden_size 100 --gpu 0 B

python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --feature_type semantics --use_tfidf --embedding_dim 4 --nhead 2  --hidden_size 100 --gpu 1 B

python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --feature_type semantics --use_tfidf --embedding_dim 4 --nhead 3  --hidden_size 100 --gpu 1 B

python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --feature_type semantics --use_tfidf --embedding_dim 4 --nhead 1  --hidden_size 32 --gpu 1 B

python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --feature_type semantics --use_tfidf --embedding_dim 4 --nhead 1  --hidden_size 200 --gpu 1 B

python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --feature_type semantics --use_tfidf --embedding_dim 4 --nhead 1  --hidden_size 300 --gpu 0 B






