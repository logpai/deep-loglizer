# unsupervised
python deeplog_demo.py --dataset BGL --feature_type sequentials --train_anomaly_ratio 0 --label_type next_log --gpu 0 B
python transformer_demo.py --dataset BGL --feature_type sequentials --train_anomaly_ratio 0 --label_type next_log  --gpu 1 B 
python ae_demo.py --dataset BGL --feature_type sequentials --train_anomaly_ratio 0 --gpu 2 B 


# supervised
python deeplog_demo.py --dataset BGL --feature_type sequentials --train_anomaly_ratio 1 --label_type anomaly --gpu 3 B
python cnn_demo.py --dataset BGL --feature_type sequentials --train_anomaly_ratio 1 --label_type anomaly --gpu 3 B