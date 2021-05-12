# unsupervised
python deeplog_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 0 --label_type next_log --gpu 0 B
python transformer_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 0 --label_type next_log  --gpu 1
python ae_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 0 --gpu 2


# supervised
python deeplog_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 0 --label_type anomaly --gpu 3
python cnn_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 0 --label_type anomaly --gpu 3