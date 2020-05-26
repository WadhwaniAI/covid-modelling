import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def evaluate(y_true, y_pred):
    err = {}
    err['mape'] = mape(y_true, y_pred)
    err['rmse'] = rmse(y_true, y_pred)
    try:
        err['rmsle'] = rmsle(y_true, y_pred)
    except:
        err['rmsle'] = None
    return err
