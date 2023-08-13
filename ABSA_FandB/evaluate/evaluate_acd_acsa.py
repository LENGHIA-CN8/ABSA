from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from loader.load_acd_acsa import aspect_sentiment_dict

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

def mi_ma_cro_f1_noNone(y_true, y_pred, bin_masks= None):
    """
    Caculate micro and macro F1 score
    :param:
        y_true: list of ground truth labels
        y_pred: list of predicting labels
        bin_masks: list of binary mask for identify targets
    output:
        micro-F1 and macro-F1 score
    """
    new_y_pred = []
    new_y_true = []
    for i in range(len(y_pred)):
        if y_pred[i] == aspect_sentiment_dict["None"] and y_true[i] == aspect_sentiment_dict["None"]:
            continue
        else:
            new_y_pred.append(y_pred[i])
            new_y_true.append(y_true[i])
    return f1_score(new_y_true, new_y_pred, average= 'micro'), f1_score(new_y_true, new_y_pred, average='macro')

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

def get_strict_acc_acd_acsa(y_true, y_pred, aspect_type_dict):
    """
    Since each term is predicted for all aspect categories ("None" label is added for
    categories that do not belong to term). strict acc (for join acd and acsa) is 1 if 
    all prediction for all categories of a term is correct else 0.
    """
    num_aspect_type = len(aspect_type_dict)
    total_cases = int(len(y_true)/num_aspect_type)
    true_cases = 0
    for i in range(total_cases):
        match = True
        for j in range(num_aspect_type):
            if y_true[i*num_aspect_type + j] != y_pred[i*num_aspect_type + j]:
                match = False
                break
        if match:
            true_cases+=1

    aspect_strict_Acc = true_cases/total_cases
    return aspect_strict_Acc

def get_strict_acc_acd(y_true, y_pred, aspect_type_dict, aspect_sentiment_dict):
    """
    Since each term is predicted for all aspect categories ("None" label is added for
    categories that do not belong to term). strict acc is 1 if prediction for all 
    unrelated categories are correct ("None") and related categories are != "None" 
    """

    num_aspect_type = len(aspect_type_dict)
    total_cases = int(len(y_true)/num_aspect_type)
    true_cases = 0
    for i in range(total_cases):
        match = True
        for j in range(num_aspect_type):
            if y_true[i*num_aspect_type + j] != y_pred[i*num_aspect_type + j]:
                if y_true[i*num_aspect_type + j] == aspect_sentiment_dict["None"]:
                    match = False
                    break
                elif y_pred[i*num_aspect_type + j] == aspect_sentiment_dict["None"]:
                    match = False
                    break
                    
        if match:
            true_cases+=1

    aspect_strict_Acc = true_cases/total_cases
    return aspect_strict_Acc
