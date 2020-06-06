
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from functools import partial
import datetime
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../../')
from main.seir.forecast import get_forecast
from main.seir.fitting import calculate_loss, train_val_split

def avg_weighted_error(region_dict, hp):
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
    if percentiles is None:
        percentiles = range(10, 100, 10), np.array([2.5, 5, 95, 97.5])
        percentiles = np.sort(np.concatenate(percentiles))
    else:
        np.sort(percentiles)
    
    ptile_dict = {}
    for ptile in percentiles:
        index_value = (df['cdf'] - ptile/100).apply(abs).idxmin()
        best_idx = df.loc[index_value - 2:index_value + 2, :]['idx'].min()
        ptile_dict[ptile] = best_idx

    return ptile_dict

def get_all_ptiles(region_dict, date_of_interest, num_evals):
    beta = find_beta(region_dict, num_evals)
    df = sort_trials(region_dict, beta, date_of_interest)
    return get_ptiles(df)

def forecast_ptiles(region_dict, deciles_idx):
    deciles_forecast = {}
    deciles_params = {}
    predictions = region_dict['m2']['predictions']
    params = region_dict['m2']['params']
    df_district = region_dict['m2']['df_district']
    df_train_nora, df_val_nora, _ = train_val_split(
        df_district, train_rollingmean=False, val_rollingmean=False, val_size=0)
    for key in deciles_idx.keys():
        deciles_forecast[key] = {}
        df_predictions = predictions[deciles_idx[key]]
        deciles_params[key] = params[deciles_idx[key]]
        deciles_forecast[key]['df_prediction'] = df_predictions
        deciles_forecast[key]['df_loss'] = calculate_loss(df_train_nora, df_val_nora, df_predictions, train_period=7,
                        which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
    return deciles_params, deciles_forecast

def set_r0_multiplier(params_dict, mul):
    new_params = params_dict.copy()
    new_params['post_lockdown_R0']= params_dict['lockdown_R0']*mul
    return new_params

def predict_r0_multipliers(region_dict, params_dict, multipliers=[0.9, 1, 1.1, 1.25], lockdown_removal_date='2020-06-01'):
    predictions_mul_dict = {}
    for mul in multipliers:
        predictions_mul_dict[mul] = {}
        predictions_mul_dict[mul]['lockdown_R0'] = mul*params_dict['lockdown_R0']
        predictions_mul_dict[mul]['df_prediction'] = get_forecast(region_dict,
            train_fit = "m2",
            best_params=set_r0_multiplier(params_dict, mul),
            lockdown_removal_date=lockdown_removal_date)    
    return predictions_mul_dict

def save_r0_mul(predictions_mul_dict, folder):
    columns_for_csv = ['date', 'total_infected', 'hospitalised', 'recovered', 'deceased']
    for (mul, val) in predictions_mul_dict.items():
        df_prediction = val['df_prediction']
        path = f'../../reports/{folder}/what-ifs/'
        if not os.path.exists(path):
            os.makedirs(path)
        df_prediction[columns_for_csv].to_csv(os.path.join(path, f'what-if-{mul}.csv'))
