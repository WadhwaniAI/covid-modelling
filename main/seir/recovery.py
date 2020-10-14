import pandas as pd
import numpy as np
from utils.fitting.loss import Loss_Calculator

def loss_parameter_recovery(param_true, param_pred, loss_function = 'rmse_log'):
    param_true = pd.DataFrame(param_true)
    param_pred = pd.DataFrame(param_pred)
    params = param_true.columns

    loss = Loss_Calculator()
    if loss_function == 'rmse':
        calculate = lambda x, y : loss._calc_rmse(x, y)
    if loss_function == 'rmse_log':
        calculate = lambda x, y : loss._calc_rmse(x, y, log=True)
    if loss_function == 'mape':
        calculate = lambda x, y : loss._calc_mape(x, y)

    loss_dict = {}
    for param in params:
        loss_dict[param] = calculate(np.array(param_true[param]), np.array(param_pred[param]))
    return loss_dict    