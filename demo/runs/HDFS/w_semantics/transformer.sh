# --embedding_dim 8
# --nhead 4
# --hidden_size 100
# --num_layers 1

# wo semantics
python transformer_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --feature_type semantics --use_tfidf B

