import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
import numpy as np
from scipy.spatial.distance import pdist, cdist
from hdbscan import HDBSCAN as dbscan
from utils.common import metrics

class Solitary_HDBSCAN():
    def __init__(self, min_cluster_size, min_samples, mode='normal-only'):
        LOG_ROOT = GET_LOGS_ROOT()
        # Dispose Loggers.
        HDBSCANLogger = logging.getLogger('Solitary_HDBSCAN')
        HDBSCANLogger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'Solitary_HDBSCAN.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        HDBSCANLogger.addHandler(console_handler)
        HDBSCANLogger.addHandler(file_handler)
        HDBSCANLogger.info(
            'Construct logger for Solitary_HDBSCAN succeeded, current working directory: %s, logs will be written in %s' %
            (os.getcwd(), LOG_ROOT))

        self.logger = HDBSCANLogger
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.model = dbscan(algorithm='best',
                            min_cluster_size=self.min_cluster_size,
                            min_samples=self.min_samples if self.min_samples != -1 else None,
                            core_dist_n_jobs=10,
                            metric='euclidean')
        self.clusters = None
        self.cluster_central = []
        self.outliers = []
        self.labels = []
        self.normal_cores = set()
        self.anomalous_cores = set()
        self.mode = mode

    def fit_predict(self, inputs):
        self.logger.info('Start training model')
        start_time = time.time()
        self.labels = self.model.fit_predict(inputs).tolist()
        self.clusters = set(self.labels)
        self.outliers = self.model.outlier_scores_.tolist()
        self.logger.info('Get Total %d clusters in %.2fs' % (len(self.clusters), (time.time() - start_time)))
        return self.labels

    def fit(self, inputs):
        self.model.fit(inputs)
        pass

    def evaluate(self, inputs, ground_truth, normal_ids, label2id):
        all_predicted = [label2id[x] for x in self.predict(inputs, normal_ids)]
        assert len(all_predicted) == len(ground_truth)
        ground_truth_without_labeled_normal = []
        predicted_label_without_labeled_normal = []
        id = 0
        for label, gt in zip(all_predicted, ground_truth):
            if id not in normal_ids:
                ground_truth_without_labeled_normal.append(gt)
                predicted_label_without_labeled_normal.append(label)
            id += 1
        precision, recall, f = metrics(predicted_label_without_labeled_normal, ground_truth_without_labeled_normal)
        self.logger.info('Precision %.4f recall %.4f f-score %.4f ' % (precision, recall, f))
        return precision, recall, f
        pass

    def min_dist(self, source, target):
        min_dist = float("inf")
        for line in target:
            d = np.linalg.norm(source[0] - line)
            if d < min_dist:
                min_dist = d
                if min_dist == 0:
                    break
        return min_dist

    def predict(self, inputs, normal_ids):
        '''
        normal_ids are involved in inputs.
        :param inputs: all input reprs
        :param normal_ids: labeled normal indexes.
        :return: predicted label for each line of inputs, labeled normal ones included.
        '''
        predicted = []
        assert len(inputs) == len(self.labels)
        inputs = np.asarray(inputs, dtype=np.float)
        self.logger.info('Summarizing labeled normals and their reprs.')
        normal_matrix = []
        for id in normal_ids:
            normal_matrix.append(inputs[id, :])
            if self.labels[id] != -1:
                self.normal_cores.add(self.labels[id])
        self.logger.info('Normal clusters are: ' + str(self.normal_cores))
        normal_matrix = np.asarray(normal_matrix, dtype=np.float)
        self.logger.info('Shape of normal matrix: %d x %d' % (normal_matrix.shape[0], normal_matrix.shape[1]))

        by_normal_core_normal = 0
        by_normal_core_anomalous = 0
        by_dist_normal = 0
        by_dist_anomalous = 0

        for id, predict_cluster in enumerate(self.labels):
            if id in normal_ids:
                # Add labeled normals as predicted normals to formalize the output format for other modules.
                predicted.append('Normal')
                continue
            if predict_cluster in self.normal_cores:
                by_normal_core_normal += 1
                predicted.append('Normal')
            elif predict_cluster == -1:
                cur_repr = inputs[id]
                dists = cdist([cur_repr], normal_matrix)
                if dists.min() == 0:
                    by_dist_normal += 1
                    predicted.append('Normal')
                else:
                    by_dist_anomalous += 1
                    predicted.append("Anomaly")

                pass
            else:
                by_normal_core_anomalous += 1
                predicted.append("Anomaly")
        self.logger.info(
            'Found %d normal, %d anomalous by normal clusters' % (by_normal_core_normal, by_normal_core_anomalous))
        self.logger.info('Found %d normal, %d anomalous by minimum distances' % (by_dist_normal, by_dist_anomalous))
        return predicted

