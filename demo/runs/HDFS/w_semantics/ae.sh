# --
# --window_size 10
# --hidden_size 200
# --num_directions 1
# --embedding_dim 8


# wo semantics
python ae_demo.py --dataset HDFS --pkl_dir ../data/processed/HDFS/hdfs_no_train_anomaly_8_2 --window_size 10 --anomaly_ratio 0.03  --feature_type semantics --use_tfidf B

