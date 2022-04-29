# PLELog 
 
 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5910349.svg)](https://doi.org/10.5281/zenodo.5910349)
 
 
This is the basic implementation of our submission in ICSE 2021: **Semi-supervised Log-based Anomaly Detection via Probabilistic Label Estimation**.
- [PLELog](#plelog)
  * [Description](#description)
  * [Project Structure](#project-structure)
  * [Datasets](#datasets)
  * [Reproducibility:](#reproducibility-)
    + [Environment:](#environment-)
    + [Preparation](#preparation)
  * [Anomaly detection](#anomaly-detection)
  * [Contact](#contact)

## Description

`PLELog` is a novel approach for log-based anomaly detection via probabilistic label estimation. 
It is designed to effectively detect anomalies in unlabeled logs and meanwhile avoid the manual labeling effort for training data generation.
We use semantic information within log events as fixed-length vectors and apply `HDBSCAN` to automatically clustering log sequences. 
After that, we also propose a Probabilistic Label Estimation approach to reduce the noises introduced by error labeling and put "labeled" instances into `attention-based GRU network` for training. 
We conducted an empirical study to evaluate the effectiveness of `PLELog` on two open-source log data (i.e., HDFS and BGL). The results demonstrate the effectiveness of `PLELog`. 
In particular, `PLELog` has been applied to two real-world systems from a university and a large corporation, further demonstrating its practicability.

## Project Structure

```
├─approaches  # PLELog main entrance.
├─config      # Configuration for Drain
├─entities    # Instances for log data and DL model.
├─utils
├─logs        
├─datasets    
├─models      # Attention-based GRU and HDBSCAN Clustering.
├─module      # Anomaly detection modules, including classifier, Attention, etc.
├─outputs           
├─parsers     # Drain parser.
├─preprocessing # preprocessing code, data loaders and cutters.
├─representations # Log template and sequence representation.
└─util        # Vocab for DL model and some other common utils.
```

## Datasets

We used `2` open-source log datasets, HDFS and BGL. 
In the future, we are planning on testing `PLELog` on more log data.

| Software System | Description                        | Time Span  | # Messages | Data Size | Link                                                      |
|       ---       |           ----                     |    ----    |    ----    |  ----     |                ---                                        |
| HDFS            | Hadoop distributed file system log | 38.7 hours | 11,175,629 | 1.47 GB   | [LogHub](https://github.com/logpai/loghub)                |
| BGL             | Blue Gene/L supercomputer log      | 214.7 days | 4,747,963  | 708.76MB  | [Usenix-CFDR Data](https://www.usenix.org/cfdr-data#hpc4) |

## Reproducibility

We have published an full version of PLELog (including HDFS log dataset, glove word embdding as well as a trained model) in Zenodo, please find the project from the zenodo badge at the beginning.

### Environment

**Note:** We attach great importance to the reproducibility of `PLELog`. To run and reproduce our results, please try to install the suggested version of the key packages.

**Key Packages:**


PyTorch v1.10.1

python v3.8.3

hdbscan v0.8.27

overrides v6.1.0

**scikit-learn v0.24**

tqdm

regex

[Drain3](https://github.com/IBM/Drain3)


hdbscan and overrides are not available while using anaconda, try using pip or:
`conda install -c conda-forge pkg==ver` where `pkg` is the target package and `ver` is the suggested version.

**Please be noted:** Since there are some known issue about joblib, scikit-learn > 0.24 is not supported here. We'll keep watching. 

### Preparation

You need to follow these steps to **completely** run `PLELog`.
- **Step 1:** To run `PLELog` on different log data, create a directory under `datasets` folder **using unique and memorable name**(e.g. HDFS and BGL). `PLELog` will try to find the related files and create logs and results according to this name.
- **Step 2:** Move target log file (plain text, each raw contains one log message) into the folder of step 1.
- **Step 3:** Download `glove.6B.300d.txt` from [Stanford NLP word embeddings](https://nlp.stanford.edu/projects/glove/), and put it under `datasets` folder.
- **Step 4:** Run `approaches/PLELog.py` (make sure it has proper parameters). You can find the details about Drain parser from [IBM](https://github.com/IBM/Drain3).


**Note:** Since log can be very different, here in this repository, we only provide the processing approach of HDFS and BGL w.r.t our experimental setting.


## Anomaly Detection

To those who are interested in applying PLELog on their log data, please refer to `BasicLoader` abstract class in preprocessing/BasicLoader.py` for more instructions.

- **Step 1:** To run `PLELog` on different log data, create a directory under `datasets` folder **using unique and memorable name**(e.g. HDFS and BGL). `PLELog` will try to find the related files and create logs and results according to this name.
- **Step 2:** Move target log file (plain text, each raw contains one log message) into the folder of step 1.
- **Step 3:** Create a new dataloader class implementing `BasicLoader`. 
- **Step 4:** Go to `preprocessing/Preprocess.py` and add your new log data into acceptable variables.

## Contact

We are happy to see `PLELog` being applied in the real world and willing to contribute to the community. Feel free to contact us if you have any question!
Authors information:

| Name          | Email Address          | 
| ------------- | ---------------------- | 
| Lin Yang      | linyang@tju.edu.cn     |
| Junjie Chen * | junjiechen@tju.edu.cn  |
| Weijing Wang  | wangweijing@tju.edu.cn |

\* *corresponding author*