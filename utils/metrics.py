import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def accuracy(pred, true):
    """分类准确率: pred/logits -> sigmoid -> >0.5 -> 与 true(0/1) 比较"""
    pred_label = (pred >= 0.5).astype(int)
    true_label = true.astype(int)
    return np.mean(pred_label == true_label)


def auc(pred, true):
    """AUC-ROC: pred 为概率值 (已过 sigmoid), true 为 0/1"""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(true.astype(int), pred)


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def classification_metric(pred_logits, true):
    """分类评估: accuracy + AUC"""
    from scipy.special import expit as sigmoid
    pred_logits = pred_logits.reshape(-1)
    true = true.reshape(-1)
    prob = sigmoid(pred_logits)
    pred_label = (prob >= 0.5).astype(int)
    true_label = true.astype(int)
    acc = np.mean(pred_label == true_label)
    try:
        from sklearn.metrics import roc_auc_score
        auc_val = roc_auc_score(true_label, prob)
    except ValueError:
        auc_val = float('nan')
    return acc, auc_val
