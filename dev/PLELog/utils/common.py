import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def metrics(y_pred, y_true):
    """ Calucate evaluation metrics for precision, recall, and f1.

    Arguments
    ---------
        y_pred: ndarry, the predicted result list
        y_true: ndarray, the ground truth label list

    Returns
    -------
        precision: float, precision value
        recall: float, recall value
        f1: float, f1 measure value
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1


def get_precision_recall(TP, TN, FP, FN):
    if TP == 0:
        return 0, 0, 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f = 2 * precision * recall / (precision + recall)
    return precision, recall, f

def not_empty(s):
    return s and s.strip()

def like_camel_to_tokens(camel_format):
    simple_format = []
    temp = ''
    flag = False

    if isinstance(camel_format, str):
        for i in range(len(camel_format)):
            if camel_format[i] == '-' or camel_format[i] == '_':
                simple_format.append(temp)
                temp = ''
                flag = False
            elif camel_format[i].isdigit():
                simple_format.append(temp)
                simple_format.append(camel_format[i])
                temp = ''
                flag = False
            elif camel_format[i].islower():
                if flag:
                    w = temp[-1]
                    temp = temp[:-1]
                    simple_format.append(temp)
                    temp = w + camel_format[i].lower()
                else:
                    temp += camel_format[i]
                flag = False
            else:
                if not flag:
                    simple_format.append(temp)
                    temp = ''
                temp += camel_format[i].lower()
                flag = True  # 需要回退
            if i == len(camel_format) - 1:
                simple_format.append(temp)
        simple_format = list(filter(not_empty, simple_format))
    return simple_format

def generate_inputs_and_labels(insts, label2idx):
    inputs = []
    labels = np.zeros(len(insts))
    for idx, inst in enumerate(insts):
        inputs.append([int(x) for x in inst.sequence])
        if inst.label in ['Normal', Anomaly]:
            if inst.label == 'Normal':
                label = 0
            else:
                label = 1
        else:
            label = int(inst.label)

        labels[idx] = label
    return inputs, labels
