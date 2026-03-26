import numpy as np

def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))

def mse(pred, true):
    return float(np.mean((pred - true) ** 2))

def rmse(pred, true):
    return float(np.sqrt(mse(pred, true)))

def mape(pred, true, eps=1e-8):
    denom = np.maximum(np.abs(true), eps)
    return float(np.mean(np.abs((pred - true) / denom)) * 100.0)