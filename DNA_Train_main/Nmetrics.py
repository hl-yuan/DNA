import numpy as np
from sklearn.metrics import  adjusted_rand_score
from sklearn import metrics
from scipy.optimize import linear_sum_assignment


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    return acc


def evaluate(truth, prediction):
    acc = cluster_acc(truth, prediction)
    nmi = metrics.normalized_mutual_info_score(truth, prediction)
    ari = adjusted_rand_score(truth, prediction)
    return tuple(float(np.round(value, 4)) for value in (acc, nmi, ari))
