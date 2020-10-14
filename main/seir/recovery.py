import pandas as pd
import numpy as np
import copy
from functools import partial

from utils.fitting.loss import Loss_Calculator
from main.seir.optimiser import Optimiser

def get_recovery_loss_parameter_wise(param_true, param_pred, param_list=[], loss_function = 'rmse_log'):
    param_true = pd.DataFrame(param_true)
    param_pred = pd.DataFrame(param_pred)

    loss = Loss_Calculator()
    if loss_function == 'rmse':
        calculate = lambda x, y : loss._calc_rmse(x, y)
    if loss_function == 'rmse_log':
        calculate = lambda x, y : loss._calc_rmse(x, y, log=True)
    if loss_function == 'mape':
        calculate = lambda x, y : loss._calc_mape(x, y)

    loss_dict = {}
    for param in param_list:
        loss_dict[param] = calculate(np.array(param_true[param]), np.array(param_pred[param]))
    return loss_dict    

def get_recovery_loss(param_true, param_pred, param_list=[]):
    x, y = [], []
    for param in param_list:
        x.append(param_true[param])
        y.append(param_pred[param])
    loss = Loss_Calculator()
    return loss._calc_rmse(np.array(x), np.array(y), log=True)

def get_top_k_with_recovery_loss(prediction_dict, param_list=[], train_fit='m1', k=10):
    trials_processed = prediction_dict[train_fit]['trials_processed']
    top_k_losses = trials_processed['losses'][:k]
    top_k_params = trials_processed['params'][:k]

    modified_default_params = copy.deepcopy(prediction_dict[train_fit]['default_params'])
    modified_default_params['starting_date'] = prediction_dict[train_fit]['df_val'].iloc[0, :]['date'].date()
    modified_default_params['observed_values'] = prediction_dict[train_fit]['df_val'].iloc[0, :]
    total_days = (prediction_dict[train_fit]['df_val'].iloc[-1, :]['date'].date() - modified_default_params['starting_date']).days
        
    partial_solve_and_compute_loss = partial(prediction_dict['m1']['optimiser'].solve_and_compute_loss,
                                                default_params=modified_default_params, total_days=total_days,
                                                loss_method='mape', loss_indices= None, df_true=prediction_dict['m1']['df_val'])

    for i,params in enumerate(top_k_params):
        top_k_params[i]['fitting val loss'] = partial_solve_and_compute_loss(params)
        top_k_params[i]['recovery loss'] = get_recovery_loss(prediction_dict['m1']['ideal_params'], params, param_list=param_list)
        top_k_params[i]['fitting train loss'] = top_k_losses[i]
    return pd.DataFrame(list(top_k_params))