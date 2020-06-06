import os
import pdb
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from hyperopt import hp, tpe, fmin, Trials
from tqdm import tqdm
from adjustText import adjust_text

from collections import OrderedDict, defaultdict
import itertools
from functools import partial
from tqdm import tqdm
import datetime
from joblib import Parallel, delayed
import copy

from data.dataloader import get_jhu_data, get_covid19india_api_data
from data.processing import get_data, get_district_time_series

from models.seir.seir_testing import SEIR_Testing
from main.seir.optimiser import Optimiser
from main.seir.losses import Loss_Calculator

from utils.enums import Columns, SEIRParams

def get_forecast(predictions_dict: dict, simulate_till=None, train_fit='m2', best_params=None, verbose=True, lockdown_removal_date=None):
    """Returns the forecasts for a given set of params of a particular geographical area

    Arguments:
        predictions_dict {dict} -- [description]

    Keyword Arguments:
        simulate_till {[type]} -- [description] (default: {None})
        train_fit {str} -- [description] (default: {'m2'})
        best_params {[type]} -- [description] (default: {None})

    Returns:
        [type] -- [description]
    """
    if verbose:
        print("getting forecasts ..")
    if simulate_till == None:
        simulate_till = datetime.datetime.strptime(predictions_dict[train_fit]['data_last_date'], '%Y-%m-%d') + datetime.timedelta(days=37)
    if best_params == None:
        best_params = predictions_dict[train_fit]['best_params']

    default_params = copy.copy(predictions_dict[train_fit]['default_params'])
    if lockdown_removal_date is not None:
        train_period = predictions_dict[train_fit]['run_params']['train_period']
        start_date = predictions_dict[train_fit]['df_train'].iloc[-train_period, :]['date']
        lockdown_removal_date = datetime.datetime.strptime(lockdown_removal_date, '%Y-%m-%d')
        default_params['lockdown_removal_day'] = (lockdown_removal_date - start_date).days
    
    df_prediction = predictions_dict[train_fit]['optimiser'].solve(best_params,
                                                                   default_params,
                                                                   predictions_dict[train_fit]['df_train'], 
                                                                   end_date=simulate_till)

    return df_prediction

def create_region_csv(predictions_dict: dict, region: str, regionType: str, icu_fraction=0.02, best_params=None):
    """Created the CSV file for one particular geographical area in the format Keshav consumes

    Arguments:
        predictions_dict {dict} -- Dict of predictions for a geographical region
        region {str} -- Region Name
        regionType {str} -- Region type ('dist', 'state')

    Keyword Arguments:
        icu_fraction {float} -- Percentage of people that are in ICU (as a fraction of active cases) (default: {0.02})
        best_params {dict} -- If not none, these params are used to get predictions, not 
        the predictions_dict['best_params'] (default: {None})

    Returns:
        pd.DataFrame -- The output CSV file in the format Keshav consumes
    """
    print("compiling csv data ..")
    columns = ['forecastRunDate', 'regionType', 'region', 'model_name', 'error_function', 'error_value', 'current_total', 'current_active', 'current_recovered',
               'current_deceased', 'current_hospitalized', 'current_icu', 'current_ventilator', 'predictionDate', 'active_mean', 'active_min',
               'active_max', 'hospitalized_mean', 'hospitalized_min', 'hospitalized_max', 'icu_mean', 'icu_min', 'icu_max', 'deceased_mean',
               'deceased_min', 'deceased_max', 'recovered_mean', 'recovered_min', 'recovered_max', 'total_mean', 'total_min', 'total_max']
    df_output = pd.DataFrame(columns=columns)

    df_prediction = get_forecast(predictions_dict, best_params=best_params)
    df_true = predictions_dict['m1']['df_district']
    prediction_daterange = np.union1d(df_true['date'], df_prediction['date'])
    no_of_data_points = len(prediction_daterange)
    df_output['predictionDate'] = prediction_daterange

    df_output['forecastRunDate'] = [datetime.datetime.today().date()]*no_of_data_points
    df_output['regionType'] = [regionType]*no_of_data_points
    df_output['region'] = [region]*no_of_data_points
    df_output['model_name'] = ['SEIR']*no_of_data_points
    df_output['error_function'] = ['MAPE']*no_of_data_points
    error = predictions_dict['m1']['df_loss'].loc['total_infected', 'val']
    df_output['error_value'] = [error]*no_of_data_points

    df_output.set_index('predictionDate', inplace=True)

    pred_hospitalisations = df_prediction['hospitalised'].to_numpy()
    error = predictions_dict['m1']['df_loss'].loc['hospitalised', 'val']
    df_output.loc[df_output.index.isin(df_prediction['date']), 'active_mean'] = pred_hospitalisations
    df_output.loc[df_output.index.isin(df_prediction['date']), 'active_min'] = (1 - 0.01*error)*pred_hospitalisations
    df_output.loc[df_output.index.isin(df_prediction['date']), 'active_max'] = (1 + 0.01*error)*pred_hospitalisations
    
    df_output.loc[df_output.index.isin(df_prediction['date']), 'hospitalized_mean'] = pred_hospitalisations
    df_output.loc[df_output.index.isin(df_prediction['date']), 'hospitalized_min'] = (1 - 0.01*error)*pred_hospitalisations
    df_output.loc[df_output.index.isin(df_prediction['date']), 'hospitalized_max'] = (1 + 0.01*error)*pred_hospitalisations
    
    df_output.loc[df_output.index.isin(df_prediction['date']), 'icu_mean'] = icu_fraction*pred_hospitalisations
    df_output.loc[df_output.index.isin(df_prediction['date']), 'icu_min'] = (1 - 0.01*error)*icu_fraction*pred_hospitalisations
    df_output.loc[df_output.index.isin(df_prediction['date']), 'icu_max'] = (1 + 0.01*error)*icu_fraction*pred_hospitalisations
    
    pred_recoveries = df_prediction['recovered'].to_numpy()
    error = predictions_dict['m1']['df_loss'].loc['recovered', 'val']
    df_output.loc[df_output.index.isin(df_prediction['date']), 'recovered_mean'] = pred_recoveries
    df_output.loc[df_output.index.isin(df_prediction['date']), 'recovered_min'] = (1 - 0.01*error)*pred_recoveries
    df_output.loc[df_output.index.isin(df_prediction['date']), 'recovered_max'] = (1 + 0.01*error)*pred_recoveries
    
    pred_fatalities = df_prediction['deceased'].to_numpy()
    error = predictions_dict['m1']['df_loss'].loc['deceased', 'val']
    df_output.loc[df_output.index.isin(df_prediction['date']), 'deceased_mean'] = pred_fatalities
    df_output.loc[df_output.index.isin(df_prediction['date']), 'deceased_min'] = (1 - 0.01*error)*pred_fatalities
    df_output.loc[df_output.index.isin(df_prediction['date']), 'deceased_max'] = (1 + 0.01*error)*pred_fatalities
    
    pred_total_cases = df_prediction['total_infected'].to_numpy()
    error = predictions_dict['m1']['df_loss'].loc['total_infected', 'val']
    df_output.loc[df_output.index.isin(df_prediction['date']), 'total_mean'] = pred_total_cases
    df_output.loc[df_output.index.isin(df_prediction['date']), 'total_min'] = (1 - 0.01*error)*pred_total_cases
    df_output.loc[df_output.index.isin(df_prediction['date']), 'total_max'] = (1 + 0.01*error)*pred_total_cases
    
    df_output.loc[df_output.index.isin(df_true['date']), 'current_total'] = df_true['total_infected'].to_numpy()
    df_output.loc[df_output.index.isin(df_true['date']), 'current_hospitalized'] = df_true['hospitalised'].to_numpy()
    df_output.loc[df_output.index.isin(df_true['date']), 'current_deceased'] = df_true['deceased'].to_numpy()
    df_output.loc[df_output.index.isin(df_true['date']), 'current_recovered'] = df_true['recovered'].to_numpy()
    df_output.reset_index(inplace=True)
    df_output = df_output[columns]
    return df_output

def create_decile_csv(decile_dict: dict, df_true: pd.DataFrame, region: str, regionType: str, icu_fraction=0.02):
    print("compiling csv data ..")
    columns = ['forecastRunDate', 'regionType', 'region', 'model_name', 'error_function', 'current_total', 'current_active', 'current_recovered',
               'current_deceased', 'current_hospitalised', 'current_icu', 'current_ventilator', 'predictionDate']
    
    for decile in decile_dict.keys():
        columns += [f'active_{decile}', f'active_{decile}_error', 
            f'hospitalised_{decile}', f'hospitalised_{decile}_error', 
            f'icu_{decile}', f'icu_{decile}_error', 
            f'recovered_{decile}', f'recovered_{decile}_error', 
            f'deceased_{decile}', f'deceased_{decile}_error', 
            f'total_{decile}', f'total_{decile}_error'
        ]

    df_output = pd.DataFrame(columns=columns)

    dateseries = decile_dict[list(decile_dict.keys())[0]]['df_prediction']['date']
    prediction_daterange = np.union1d(df_true['date'], dateseries)
    no_of_data_points = len(prediction_daterange)
    df_output['predictionDate'] = prediction_daterange

    df_output['forecastRunDate'] = [datetime.datetime.today().date()]*no_of_data_points
    df_output['regionType'] = [regionType]*no_of_data_points
    df_output['region'] = [region]*no_of_data_points
    df_output['model_name'] = ['SEIR']*no_of_data_points
    df_output['error_function'] = ['MAPE']*no_of_data_points
    df_output.set_index('predictionDate', inplace=True)

    for decile in decile_dict.keys():
        df_prediction = decile_dict[decile]['df_prediction']
        df_prediction = df_prediction.set_index('date')
        df_loss = decile_dict[decile]['df_loss']
        df_output.loc[df_prediction.index, f'active_{decile}'] = df_prediction['hospitalised']
        df_output.loc[df_prediction.index, f'active_{decile}_error'] = df_loss.loc['hospitalised', 'train']
        df_output.loc[df_prediction.index, f'hospitalised_{decile}'] = df_prediction['hospitalised']
        df_output.loc[df_prediction.index, f'hospitalised_{decile}_error'] = df_loss.loc['hospitalised', 'train']
        df_output.loc[df_prediction.index, f'icu_{decile}'] = icu_fraction*df_prediction['hospitalised']
        df_output.loc[df_prediction.index, f'icu_{decile}_error'] = df_loss.loc['hospitalised', 'train']
        df_output.loc[df_prediction.index, f'recovered_{decile}'] = df_prediction['recovered']
        df_output.loc[df_prediction.index, f'recovered_{decile}_error'] = df_loss.loc['recovered', 'train']
        df_output.loc[df_prediction.index, f'deceased_{decile}'] = df_prediction['deceased']
        df_output.loc[df_prediction.index, f'deceased_{decile}_error'] = df_loss.loc['deceased', 'train']
        df_output.loc[df_prediction.index, f'total_{decile}'] = df_prediction['total_infected']
        df_output.loc[df_prediction.index, f'total_{decile}_error'] = df_loss.loc['total_infected', 'train']

    df_true = df_true.set_index('date')
    df_output.loc[df_true.index, 'current_total'] = df_true['total_infected'].to_numpy()
    df_output.loc[df_true.index, 'current_hospitalised'] = df_true['hospitalised'].to_numpy()
    df_output.loc[df_true.index, 'current_deceased'] = df_true['deceased'].to_numpy()
    df_output.loc[df_true.index, 'current_recovered'] = df_true['recovered'].to_numpy()
    
    df_output.reset_index(inplace=True)
    # df_output = df_output[columns]
    return df_output

def create_all_csvs(predictions_dict: dict, icu_fraction=0.02):
    """Creates the output for all geographical regions (not just one)

    Arguments:
        predictions_dict {dict} -- The predictions dict for all geographical regions

    Keyword Arguments:
        icu_fraction {float} -- Percentage of active cases that are in the ICU (default: {0.02})

    Returns:
        pd.DataFrame -- output for all geographical regions
    """
    columns = ['forecastRunDate', 'regionType', 'region', 'model_name', 'error_function', 'error_value', 'current_total', 'current_active', 'current_recovered',
               'current_deceased', 'current_hospitalized', 'current_icu', 'current_ventilator', 'predictionDate', 'active_mean', 'active_min',
               'active_max', 'hospitalized_mean', 'hospitalized_min', 'hospitalized_max', 'icu_mean', 'icu_min', 'icu_max', 'deceased_mean',
               'deceased_min', 'deceased_max', 'recovered_mean', 'recovered_min', 'recovered_max', 'total_mean', 'total_min', 'total_max']
    df_final = pd.DataFrame(columns=columns)
    for region in predictions_dict.keys():
        if region[1] == None:
            df_output = create_region_csv(predictions_dict[region], region=region[0], regionType='state', 
                                          icu_fraction=icu_fraction)
        else:
            df_output = create_region_csv(predictions_dict[region], region=region[1], regionType='district',
                                        icu_fraction=icu_fraction)
        df_final = pd.concat([df_final, df_output], ignore_index=True)
    
    return df_final

def write_csv(df_final: pd.DataFrame, filename:str=None):
    """Helper function for saving the CSV files

    Arguments:
        df_final {pd.DataFrame} -- the final CSV to be saved
        filename {str} -- the name of the file
    """
    if filename == None:
        filename = '../../output-{}.csv'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    df_final.to_csv(filename, index=False)

def order_trials(m_dict: dict):
    params_array = []
    for trial in m_dict['trials']:
        params_dict = copy.copy(trial['misc']['vals'])
        for key in params_dict.keys():
            params_dict[key] = params_dict[key][0]
        params_array.append(params_dict)
    params_array = np.array(params_array)
    losses_array = np.array([trial['result']['loss'] for trial in m_dict['trials']])
    
    least_losses_indices = np.argsort(losses_array)
    losses_array = losses_array[least_losses_indices]
    params_array = params_array[least_losses_indices]
    return params_array, losses_array

def top_k_trials(m_dict: dict, k=10):
    params_array, losses_array = order_trials(m_dict)
    return losses_array[:k], params_array[:k]

def forecast_k(predictions_dict: dict, k=10, train_fit='m2', forecast_days=37):
    top_k_losses, top_k_params = top_k_trials(predictions_dict[train_fit], k=k)
    predictions = []
    dots = ['.']
    simulate_till = datetime.datetime.strptime(predictions_dict[train_fit]['data_last_date'], '%Y-%m-%d') + datetime.timedelta(days=forecast_days)
    print("getting forecasts ..")
    for i, params_dict in tqdm(enumerate(top_k_params)):
        predictions.append(get_forecast(
            predictions_dict, best_params=params_dict, train_fit=train_fit, simulate_till=simulate_till, verbose=False))
    return predictions, top_k_losses, top_k_params


def get_all_trials(predictions_dict, train_fit='m2', forecast_days=37):
    predictions, losses, params = forecast_k(
        predictions_dict, 
        k=len(predictions_dict[train_fit]['trials']), 
        train_fit=train_fit,
        forecast_days=forecast_days
    )
    return predictions, losses, params
