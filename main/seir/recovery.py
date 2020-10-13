import pandas as pd
from utils.fitting.loss import Loss_Calculator

def loss_parameter_recovery(param_true, param_pred, loss_function = 'rsme'):
    df = pd.DataFrame([param_true, param_pred])
    
    loss = Loss_Calculator()
    if (loss_function == 'rsme'):
        return loss._calc_rmse(df.iloc[1,:], df.iloc[0,:])
    elif (loss_function == 'rmse_log'):
        return loss._calc_rmse(df.iloc[1,:], df.iloc[0,:], log=True)
    elif (loss_function == 'mape'):
        return loss._calc_mape(df.iloc[1,:], df.iloc[0,:])
    