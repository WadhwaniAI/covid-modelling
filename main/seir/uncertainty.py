
import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
from functools import partial
from hyperopt import fmin, tpe, hp, Trials

sys.path.append('../../')
from main.seir.forecast import get_forecast
from main.seir.fitting import calculate_loss, train_val_split

def avg_weighted_error(region_dict, hp):
    """
    Loss function to optimize beta

    Args:
        region_dict (dict): region_dict as returned by main.seir.fitting.single_fitting_cycle
        hp (dict): {'beta': float}

    Returns:
        float: average relative error calculated over trials and a val set
    """    
    beta = hp['beta']
    losses = region_dict['m1']['losses']
    df_val = region_dict['m1']['df_district'].set_index('date') \
        .loc[region_dict['m1']['df_val']['date'],:]
    active_predictions = region_dict['m1']['all_trials'].loc[:, df_val.index]
    beta_loss = np.exp(-beta*losses)
    avg_rel_err = 0
    for date in df_val.index:
        weighted_pred = (beta_loss*active_predictions[date]).sum() / beta_loss.sum()
        rel_error = (weighted_pred - df_val.loc[date,'hospitalised']) / df_val.loc[date,'hospitalised']
        avg_rel_err += rel_error
    avg_rel_err /= len(df_val)
    return avg_rel_err

def find_beta(region_dict, num_evals=1000):
    """
    Runs a search over m1 trials to find best beta for a probability distro

    Args:
        num_evals (int, optional): number of iterations to run hyperopt. Defaults to 1000.

    Returns:
        float: optimal beta value
    """    
    searchspace = {
        'beta': hp.uniform('beta', 0, 10)
    }
    trials = Trials()
    best = fmin(partial(avg_weighted_error, region_dict),
                space=searchspace,
                algo=tpe.suggest,
                max_evals=num_evals,
                trials=trials)

    return best['beta']

def sort_trials(region_dict, beta, date_of_interest):
    """
    Computes probability distribution based on given beta and date 
    over the trials in region_dict['m2']['all_trials']

    Args:
        region_dict (dict): region_dict as returned by main.seir.fitting.single_fitting_cycle
        beta (float): computed beta value for distribution
        date_of_interest (str): prediction date by which trials should be sorted + distributed

    Returns:
        pd.DataFrame: dataframe of sorted trials, with columns
            idx: original trial index
            loss: loss value for that trial
            weight: np.exp(-beta*loss)
            pdf: pdf
            cdf: cdf
            <date_of_interest>: predicted value on <date_of_interest>

    """    
    date_of_interest = datetime.datetime.strptime(date_of_interest, '%Y-%m-%d')
    
    df = pd.DataFrame(columns=['loss', 'weight', 'pdf', date_of_interest, 'cdf'])
    df['loss'] = region_dict['m2']['losses']
    df['weight'] = np.exp(-beta*df['loss'])
    df['pdf'] = df['weight'] / df['weight'].sum()
    df[date_of_interest] = region_dict['m2']['all_trials'].loc[:, date_of_interest]
    
    df = df.sort_values(by=date_of_interest)
    df.index.name = 'idx'
    df.reset_index(inplace=True)
    
    df['cdf'] = df['pdf'].cumsum()
    
    return df

def get_ptiles(df, percentiles=None):
    """
    Get the predictions at certain percentiles from a distribution of trials

    Args:
        df (pd.DataFrame): the probability distribution of the trials as returned by sort_trials
        percentiles (list, optional): percentiles at which predictions from the distribution 
            will be returned. Defaults to all deciles 10-90, as well as 2.5/97.5 and 5/95.

    Returns:
        dict: {percentile: index} where index is the trial index (to arrays in predictions_dict)
    """    
    if percentiles is None:
        percentiles = range(10, 100, 10), np.array([2.5, 5, 95, 97.5])
        percentiles = np.sort(np.concatenate(percentiles))
    else:
        np.sort(percentiles)
    
    ptile_dict = {}
    for ptile in percentiles:
        index_value = (df['cdf'] - ptile/100).apply(abs).idxmin()
        best_idx = df.loc[index_value - 2:index_value + 2, :]['idx'].min()
        ptile_dict[ptile] = int(best_idx)

    return ptile_dict

def get_all_ptiles(region_dict, date_of_interest, num_evals, percentiles=None):
    """
    Computes probability distribution and returns 

    Args:
        region_dict (dict): region_dict as returned by main.seir.fitting.single_fitting_cycle
        date_of_interest (str): prediction date by which trials should be sorted + distributed
        num_evals (int): number of iteratiosn hyperopt should run to find beta value

    Returns:
        tuple: (dict, float)
            dict: {percentile: index} where index is the trial index (to arrays in predictions_dict)
            float: beta value used to create distribution
    """    
    beta = find_beta(region_dict, num_evals)
    df = sort_trials(region_dict, beta, date_of_interest)
    return get_ptiles(df, percentiles=percentiles), beta

def forecast_ptiles(region_dict, ptile_dict):
    """
    Get forecasts at certain percentiles

    Args:
        region_dict (dict): region_dict as returned by main.seir.fitting.single_fitting_cycle
        ptile_dict (dict): {percentile: trial index} as returned by get_ptiles/get_all_ptiles

    Returns:
        tuple: deciles_params, deciles_forecast
            dict: deciles_params, {percentile: params}
            dict: deciles_forecast, {percentile: {df_prediction: pd.DataFrame, df_loss: pd.DataFrame}}
    """    
    deciles_forecast = {}
    deciles_params = {}
    predictions = region_dict['m2']['predictions']
    params = region_dict['m2']['params']
    df_district = region_dict['m2']['df_district']
    df_train_nora = df_district.set_index('date').loc[region_dict['m2']['df_train']['date'],:].reset_index()
    for key in ptile_dict.keys():
        deciles_forecast[key] = {}
        df_predictions = predictions[ptile_dict[key]]
        deciles_params[key] = params[ptile_dict[key]]
        deciles_forecast[key]['df_prediction'] = df_predictions
        deciles_forecast[key]['df_loss'] = calculate_loss(df_train_nora, None, df_predictions, train_period=7,
                        which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
    return deciles_params, deciles_forecast

def set_r0_multiplier(params_dict, mul):
    """[summary]

    Args:
        params_dict (dict): model parameters
        mul (float): float to multiply lockdown_R0 by to get post_lockdown_R0

    Returns:
        dict: model parameters with a post_lockdown_R0
    """    
    new_params = params_dict.copy()
    new_params['post_lockdown_R0']= params_dict['lockdown_R0']*mul
    return new_params

def predict_r0_multipliers(region_dict, params_dict, multipliers=[0.9, 1, 1.1, 1.25], lockdown_removal_date='2020-06-01'):
    """
    Function to predict what-if scenarios with different post-lockdown R0s

    Args:
        region_dict (dict): region_dict as returned by main.seir.fitting.single_fitting_cycle
        params_dict (dict): model parameters
        multipliers (list, optional): list of multipliers to get post_lockdown_R0 from lockdown_R0. 
            Defaults to [0.9, 1, 1.1, 1.25].
        lockdown_removal_date (str, optional): Date to change R0 value and simulate change. 
            Defaults to '2020-06-01'.

    Returns:
        dict: {
            multiplier: {
                params: multiplied params dict, 
                df_prediction: predictions
            }
        }
    """    
    predictions_mul_dict = {}
    for mul in multipliers:
        predictions_mul_dict[mul] = {}
        new_params = set_r0_multiplier(params_dict, mul)
        predictions_mul_dict[mul]['params'] = new_params
        predictions_mul_dict[mul]['df_prediction'] = get_forecast(region_dict,
            train_fit = "m2",
            best_params=new_params,
            lockdown_removal_date=lockdown_removal_date)    
    return predictions_mul_dict

def save_r0_mul(predictions_mul_dict, folder):
    """
    Saves what-if scenario plots and csv data

    Args:
        predictions_mul_dict (dict): output from predict_r0_multipliers 
            {multiplier: {params: dict, df_predicted: pd.DataFrame}}
        folder (str): assets will be saved in reports/{folder}/ 
    """    
    columns_for_csv = ['date', 'total_infected', 'hospitalised', 'recovered', 'deceased']
    for (mul, val) in predictions_mul_dict.items():
        df_prediction = val['df_prediction']
        path = f'../../reports/{folder}/what-ifs/'
        if not os.path.exists(path):
            os.makedirs(path)
        df_prediction[columns_for_csv].to_csv(os.path.join(path, f'what-if-{mul}.csv'))
    pd.DataFrame({key: val['params'] for key, val in predictions_mul_dict.items()}) \
        .to_csv(f'../../reports/{folder}/what-ifs/what-ifs-params.csv')
