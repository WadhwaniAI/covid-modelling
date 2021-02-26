import numpy as np
import pandas as pd


def get_best_param_dist(model_dict):
    """Computes mean and variance of the best fit params accross different runs

    Args:
        model_dict (dict): Dict containing the predictions dict for all the runs for a given 
            scenario, config setting
        
    Returns:
        dataframe containing mean, std for all the parameters 
    """
    best_param_vals = [run_dict['best_params'] for _, run_dict in model_dict.items()]
    df = pd.DataFrame(best_param_vals).describe()
    return df.loc[['mean', 'std']]

def get_ensemble_combined(model_dict, weighting='exp', beta=1):
    """Computes ensemble mean and variance of all the params accross different runs

    Args:
        model_dict (dict): Dict containing the predictions dict for all the runs for a given 
            scenario, config setting
        weighting (str, optional): The weighting function. 
            If 'exp', np.exp(-beta*loss) is the weighting function used. (beta is separate param here)
            If 'inv', 1/loss is used. Else, uniform weighting is used. Defaults to 'exp'.
        beta (float, optional): beta param for exponential weighting 

    Returns:
        dataframe containing mean, std for all the parameters 
    """
    params_dict = { k: np.array([]) for k in model_dict[list(model_dict.keys())[0]]['best_params'].keys() }
    losses_array = np.array([])
    for _, run_dict in model_dict.items():
        params_array = run_dict['trials']['params']
        loss_array = run_dict['trials']['losses']
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
    
    if 'beta' not in params_dict and 'lockdown_R0' in params_dict and 'T_inc' in params_dict:
        params_dict['beta'] = np.divide(params_dict['lockdown_R0'], params_dict['T_inc'])
    param_dist_stats = {}
    for param in params_dict.keys():
        mean = np.average(params_dict[param], weights=weights)
        variance = np.average((params_dict[param] - mean)**2, weights=weights)
        param_dist_stats[param] = {'mean':mean, 'std':np.sqrt(variance)}
    
    df = pd.DataFrame(param_dist_stats)
    return df.loc[['mean','std']]

def get_param_stats(model_dict, method='best', weighting='exp'):
    """Computes mean and variance for all the params accross different runs based on the method mentioned

    Args:
        model_dict (dict): Dict containing the predictions dict for all the runs for a given 
            scenario, config setting
        method (str, optional): The method of aggregation of different runs ('best' or 'ensemble')
        weighting (str, optional): The weighting function. 
            If 'exp', np.exp(-beta*loss) is the weighting function used. (beta is separate param here)
            If 'inv', 1/loss is used. Else, uniform weighting is used. Defaults to 'exp'.
        
    Returns:
        dataframe containing mean, std for all the parameters 
    """
    if method == 'best':
        return get_best_param_dist(model_dict)
    elif method == 'ensemble':
        return get_ensemble_combined(model_dict, weighting=weighting)

def get_loss_stats(model_dict, which_loss='train',method='best_loss_nora',weighting='exp',beta=1.0):
    """Computes mean and variance of loss values accross all the compartments for different runs

    Args:
        model_dict (dict): Dict containing the predictions dict for all the runs for a given 
            scenario, config setting
        which_losses: Which losses have to considered? train or val
        method (str, optional): The method of aggregation of different runs. 
            possible values: 'best_loss_nora', 'best_loss_ra', 'ensemble_loss_ra'
        weighting (str, optional): The weighting function. 
            If 'exp', np.exp(-beta*loss) is the weighting function used. (beta is separate param here)
            If 'inv', 1/loss is used. Else, uniform weighting is used. Defaults to 'exp'.
        
    Returns:
        dataframe containing mean, std loss values for all the compartments  
    """
    if method == 'best_loss_nora':
        loss_vals = []
        for _, run_dict in model_dict.items():
            df = run_dict['df_loss'][which_loss]
            df['agg'] = df.mean()
            loss_vals.append(df)
        df = pd.DataFrame(loss_vals).describe()
        return df.loc[['mean','std']]

    elif method == 'best_loss_ra':
        losses_array = np.array([])
        for _, run_dict in model_dict.items():
            loss_array = run_dict['trials']['losses']
            losses_array = np.append(losses_array, min(loss_array))
        df = pd.DataFrame(columns=['agg'],index=['mean','std'])
        df['agg']['mean'] = np.mean(losses_array)
        df['agg']['std'] = np.std(losses_array)
        return df
    
    elif method == 'ensemble_loss_ra':
        losses_array = np.array([])
        for _, run_dict in model_dict.items():
            loss_array = run_dict['trials']['losses']
            losses_array = np.concatenate((losses_array, loss_array), axis=0)
        df = pd.DataFrame(columns=['agg'],index=['mean','std'])
        if weighting == 'exp':
            weights = np.exp(-beta*np.array(losses_array))
        elif weighting == 'inverse':
            weights = 1/np.array(losses_array)
        else:
            weights = np.ones(np.array(losses_array).shape)
        
        mean = np.average(losses_array, weights=weights)
        variance = np.average((losses_array - mean)**2, weights=weights)
        df['agg']['mean'] = mean
        df['agg']['std'] = np.sqrt(variance)
        return df
