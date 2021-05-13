# unsupervised
python deeplog_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 0 --label_type next_log --gpu 0 B
python transformer_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 0 --label_type next_log  --gpu 1 B 
python ae_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 0 --gpu 2 B 

# supervised
python deeplog_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 1 --label_type anomaly --gpu 3 B
python cnn_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 1 --label_type anomaly --gpu 3 B





--embedding_dim 8 --hidden_size 100
--embedding_dim 8 --hidden_size 200
--embedding_dim 8 --hidden_size 300
--embedding_dim 8 --hidden_size 400
--embedding_dim 8 --hidden_size 500


python deeplog_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 0 --label_type next_log --gpu 0 --embedding_dim 8 --hidden_size 32 B
python deeplog_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 0 --label_type next_log --gpu 0 --embedding_dim 8 --hidden_size 100 B
python deeplog_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 0 --label_type next_log --gpu 0 --embedding_dim 8 --hidden_size 200 B
python deeplog_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 0 --label_type next_log --gpu 0 --embedding_dim 8 --hidden_size 300 B
python deeplog_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 0 --label_type next_log --gpu 0 --embedding_dim 8 --hidden_size 400 B
python deeplog_demo.py --dataset HDFS --feature_type sequentials --train_anomaly_ratio 0 --label_type next_log --gpu 0 --embedding_dim 8 --hidden_size 500 B