import numpy as np
import scipy.io as sio
from scipy import sparse

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics import normalized_mutual_info_score as nmi_ori
from sklearn.metrics import accuracy_score

def loadmat(path, to_dense=True):
    data = sio.loadmat(path)
    X = data["X"]
    y_true = data["y_true"].astype(np.int32).reshape(-1)

    if sparse.isspmatrix(X) and to_dense:
        X = X.toarray()

    N, dim, c_true = X.shape[0], X.shape[1], len(np.unique(y_true))
    X = X.astype(np.float64)
    return X, y_true, N, dim, c_true

def evaluate_performance(y_true, y_pred):
    acc_ = accuracy(y_true, y_pred)
    nmi_ = nmi(y_true, y_pred)
    purity_ = purity(y_true, y_pred)
    return acc_, nmi_, purity_

def accuracy(y_true, y_pred):
    """Get the best accuracy.

    Parameters
    ----------
    y_true: array-like
        The true row labels, given as external information
    y_pred: array-like
        The row labels predicted by the model

    Returns
    -------
    float
        Best value of accuracy
    """

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cost_m = np.max(cm) - cm
    indices = linear_sum_assignment(cost_m)
    indices = np.asarray(indices)
    indexes = np.transpose(indices)
    total = 0
    for row, column in indexes:
        value = cm[row][column]
        total += value
    return total * 1. / np.sum(cm)

def nmi(y_true, y_pred, average_method="max"):
    ret = nmi_ori(labels_true=y_true, labels_pred=y_pred, average_method=average_method)
    return ret

def purity(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)