import pandas as pd
from utils.fitting.loss import Loss_Calculator

def loss_parameter_recovery(param_true, param_pred, loss_function = 'rsme'):
    df = pd.DataFrame([param_true, param_pred])
    
    loss = Loss_Calculator()
    if loss_function == 'rmse':
        calculate = lambda x, y : loss._calc_rmse(x, y)
    if loss_function == 'rmse_log':
        calculate = lambda x, y : loss._calc_rmse(x, y, log=True)
    if loss_function == 'mape':
        calculate = lambda x, y : loss._calc_mape(x, y)

    loss_dict = {}
    for i, param in enumerate(df.columns):
        loss_dict[param] = calculate(df.iloc[1,i], df.iloc[0,i])
    return loss_dict    