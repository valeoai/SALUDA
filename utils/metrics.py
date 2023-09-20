import numpy as np


def ignore_cm_adaption(cm, ignore_list):
    for idx in ignore_list: 
        #set the row and the column of the idx to 0
        cm[idx,:] = 0
        cm[:,idx] = 0
    return cm 

def stats_overall_accuracy(cm, ignore_list=[]):
    """Computes the overall accuracy.

    # Arguments:
        cm: 2-D numpy array.
            Confusion matrix.
    """
    if len(ignore_list) > 0: 
        #Set to 0 the rows and columns of the ignored class
        cm = ignore_cm_adaption(cm, ignore_list)
    return np.trace(cm) / cm.sum()


def stats_pfa_per_class(cm,ignore_list=[]):
    """Computes the probability of false alarms.

    # Arguments:
        cm: 2-D numpy array.
            Confusion matrix.
    """
    if len(ignore_list) > 0: 
        #Set to 0 the rows and columns of the ignored class
        cm = ignore_cm_adaption(cm, ignore_list)
    sums = np.sum(cm, axis=0)
    mask = sums > 0
    sums[sums == 0] = 1
    pfa_per_class = (cm.sum(axis=0) - np.diag(cm)) / sums
    pfa_per_class[np.logical_not(mask)] = -1
    average_pfa = pfa_per_class[mask].mean()
    return average_pfa, pfa_per_class


def stats_accuracy_per_class(cm, ignore_list=[]):
    """Computes the accuracy per class and average accuracy.

    # Arguments:
        cm: 2-D numpy array.
            Confusion matrix.
        ignore_list: list of classes that are ignored in calculation

    # Returns
        average_accuracy: float.
            The average accuracy.
        accuracy_per_class: 1-D numpy array.
            The accuracy per class.
    """
    if len(ignore_list) > 0: 
        #Set to 0 the rows and columns of the ignored class
        cm = ignore_cm_adaption(cm, ignore_list)
    sums = np.sum(cm, axis=1)
    mask = sums > 0
    sums[sums == 0] = 1
    accuracy_per_class = np.diag(cm) / sums  # sum over lines
    accuracy_per_class[np.logical_not(mask)] = -1
    average_accuracy = accuracy_per_class[mask].mean()
    return average_accuracy, accuracy_per_class


def stats_iou_per_class(cm, ignore_list=[] ):
    """Computes the IoU per class and average IoU.

    # Arguments:
        cm: 2-D numpy array.
            Confusion matrix.

    # Returns
        average_accuracy: float.
            The average IoU.
        accuracy_per_class: 1-D numpy array.
            The IoU per class.
    """
    if len(ignore_list) > 0: 
        #Set to 0 the rows and columns of the ignored class
        cm = ignore_cm_adaption(cm, ignore_list)
    
    # compute TP, FN et FP
    TP = np.diagonal(cm, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(cm, axis=-1)
    TP_plus_FP = np.sum(cm, axis=-2)

    # compute IoU
    mask = TP_plus_FN == 0
    IoU = TP / (TP_plus_FN + TP_plus_FP - TP + mask)

    # replace IoU with 0 by the average IoU
    aIoU = IoU[np.logical_not(mask)].mean(axis=-1, keepdims=True)
    IoU += mask * aIoU

    return IoU.mean(axis=-1), IoU


def stats_f1score_per_class(cm, ignore_list=[]):
    """Computes the F1 per class and average F1.

    # Arguments:
        cm: 2-D numpy array.
            Confusion matrix.

    # Returns
        average_accuracy: float.
            The average F1.
        accuracy_per_class: 1-D numpy array.
            The F1 per class.
    """
    if len(ignore_list) > 0: 
        #Set to 0 the rows and columns of the ignored class
        cm = ignore_cm_adaption(cm, ignore_list)
    # defined as 2 * recall * prec / recall + prec
    sums = np.sum(cm, axis=1) + np.sum(cm, axis=0)
    mask = sums > 0
    sums[sums == 0] = 1
    f1score_per_class = 2 * np.diag(cm) / sums
    f1score_per_class[np.logical_not(mask)] = -1
    average_f1_score = f1score_per_class[mask].mean()
    return average_f1_score, f1score_per_class
