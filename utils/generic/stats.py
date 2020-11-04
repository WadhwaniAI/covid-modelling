import pandas as pd
import numpy as np
from main.seir.forecast import _order_trials_by_loss

def get_best_param_dist(model_dict):
    best_param_vals = []
    for _, run_dict in model_dict.items():
        best_param_vals.append(run_dict['best_params'])
    df = pd.DataFrame(best_param_vals).describe()
    return df.loc[['mean','std']]

def get_ensemble_combined(model_dict, weighting='exp', beta=1):
    params_dict = { k: np.array([]) for k in model_dict[list(model_dict.keys())[0]]['best_params'].keys() }
    losses_array = np.array([])
    for _, run_dict in model_dict.items():
        params_array, loss_array = _order_trials_by_loss(run_dict)
        losses_array = np.concatenate((losses_array, loss_array), axis=0)
        for param in params_dict.keys():
            params_vals = np.array([param_dict[param] for param_dict in params_array])
            params_dict[param] = np.concatenate((params_dict[param],params_vals),axis=0)
    
    if weighting == 'exp':
        weights = np.exp(-beta*np.array(losses_array))
    elif weighting == 'inverse':
        weights = 1/np.array(losses_array)
    else:
        weights = np.ones(np.array(losses_array).shape)
    
    param_dist_stats = {}
    for param in params_dict.keys():
        mean = np.average(params_dict[param], weights=weights)
        variance = np.average((params_dict[param] - mean)**2, weights=weights)
        param_dist_stats[param] = {'mean':mean, 'std':np.sqrt(variance)}
    
    df = pd.DataFrame(param_dist_stats)
    return df.loc[['mean','std']]

def get_param_stats(model_dict, method='best', weighting=None):
    if method == 'best':
        return get_best_param_dist(model_dict)
    elif method == 'ensemble_combined':
        return get_ensemble_combined(model_dict, weighting=weighting)

def get_loss_stats(model_dict, which_loss='train'):
    loss_vals = []
    for _, run_dict in model_dict.items():
        df = run_dict['df_loss'][which_loss]
        df['agg'] = df.mean()
        loss_vals.append(df)
    df = pd.DataFrame(loss_vals).describe()
    return df.loc[['mean','std']]