# Deep-loglizer

**Deep-loglizer is a deep learning-based log analysis toolkit for automated anomaly detection.**

If you use deep-loglizer in your research for publication, please kindly cite the following paper:

- Zhuangbin Chen, Jinyang Liu, Wenwei Gu, Yuxin Su, and Michael R. Lyu. [Experience Report: Deep Learning-based System Log Analysis for Anomaly Detection.](https://arxiv.org/abs/2107.05908) arXiv preprint, arXiv:2107.05908 (2021).


## Framework

![Deep Learning-based Log Anomaly Detection](./docs/log_ad.jpg)


## Models

| Model | Paper reference |
| :--- | :--- |
| **Unsupervised models** | |
| LSTM | [CSS'17] [Deeplog: Anomaly detection and diagnosis from system logs through deep learning](https://dl.acm.org/doi/abs/10.1145/3133956.3134015), by Min Du, Feifei Li, Guineng Zheng, and Vivek Srikumar. [University of Utah] |
| LSTM | [IJCAI'19] [LogAnomaly: unsupervised detection of sequential and quantitative anomalies in unstructured logs](https://www.ijcai.org/proceedings/2019/658) by Weibin Meng, Ying Liu, Yichen Zhu et al. [Tsinghua University] |
| Transformer | [ICDM'20] [Self-attentive classification-based anomaly detection in unstructured logs](https://ieeexplore.ieee.org/document/9338283), by Sasho Nedelkoski, Jasmin Bogatinovski, Alexander Acker, Jorge Cardoso, and Odej Kao. [TU Berlin] |
| Autoencoder | [ICT Express'20] [Unsupervised log message anomaly detection](https://www.sciencedirect.com/science/article/pii/S2405959520300643), by Amir Farzad and T Aaron Gulliver. [University of Victoria] |
| **Supervised models** | |
| Attentional BiLSTM| [ESEC/FSE'19] [Robust log-based anomaly detection on unstable log data](https://dl.acm.org/doi/10.1145/3338906.3338931) by Xu Zhang, Yong Xu, Qingwei Lin et al. [MSRA]|
| CNN | [DASC'18] [Detecting anomaly in big data system logs using convolutional neural network](https://ieeexplore.ieee.org/document/8511880) by Siyang Lu, Xiang Wei, Yandong Li, and Liqiang Wang. [University of Central Florida] |

## Install

```bash
git clone https://github.com/logpai/deep-loglizer.git
cd deep-loglizer
pip install -r requirements.txt
```

## Contributors

- [Zhuangbin Chen](http://www.cse.cuhk.edu.hk/~zbchen), The Chinese University of Hong Kong
- [Jinyang Liu](http://www.cse.cuhk.edu.hk/~jyliu), The Chinese University of Hong Kong
- Wenwei Gu, The Chinese University of Hong Kong
