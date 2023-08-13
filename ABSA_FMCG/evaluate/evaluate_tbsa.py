from sklearn.metrics import f1_score, confusion_matrix
import numpy as np



def mi_ma_cro_f1(y_true, y_pred, bin_masks= None):
    """
    Caculate micro and macro F1 score
    :param:
        y_true: list of ground truth labels
        y_pred: list of predicting labels
        bin_masks: list of binary mask for identify targets
    output:
        micro-F1 and macro-F1 score
    """
    if bin_masks is None:
        return f1_score(y_true, y_pred, average= 'micro'), f1_score(y_true, y_pred, average='macro')

    groundtruth = []
    prediction = []
    for i, check in enumerate(bin_masks):
        if check == 1:
            groundtruth.append(y_true[i])
            prediction.append(y_pred[i])
    return f1_score(groundtruth, prediction, average='micro'), f1_score(groundtruth, prediction, average='macro')

def get_confusion_matrix(y_true, y_pred, label_dict, bin_masks= None):
    """
    Caculate confusion matrix
    :param:
        y_true: list of ground truth labels
        y_pred: list of predicting labels
        bin_masks: list of binary mask for identify targets
    output:
        confusion matrix
    """
    if bin_masks is None:
        return confusion_matrix(y_true, y_pred, labels= sorted(list(label_dict.values())))
    groundtruth = []
    prediction = []
    for i, check in enumerate(bin_masks):
        if check == 1:
            groundtruth.append(y_true[i])
            prediction.append(y_pred[i])
    return confusion_matrix(groundtruth, prediction, labels= sorted(label_dict.values()))