import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

from collections import OrderedDict, defaultdict
import itertools
from functools import partial
from tqdm import tqdm
import datetime
from joblib import Parallel, delayed
import copy

from data.processing.whatifs import scale_up_acc_to_testing
from main.seir.fitting import *
from models.seir import SEIRHD, SEIR_Movement, SEIR_Movement_Testing, SEIR_Testing
from main.seir.optimiser import Optimiser

from utils.enums import Columns, SEIRParams

def get_forecast(predictions_dict: dict, days: int=37, simulate_till=None, train_fit='m2', model=SEIRHD,
                 best_params=None, verbose=True, lockdown_removal_date=None):
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
        simulate_till = datetime.datetime.strptime(predictions_dict[train_fit]['data_last_date'], '%Y-%m-%d') + \
            datetime.timedelta(days=days)
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
                                                                   model=model,
                                                                   end_date=simulate_till)

    return df_prediction


def create_region_csv(predictions_dict: dict, region: str, regionType: str, model=SEIRHD, df_prediction=None,
                      icu_fraction=0.02, best_params=None, days=30):
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
    columns = ['forecastRunDate', 'predictionDate', 'regionType', 'region', 'model_name', 'error_function', 'error_value',
                'current_total', 'current_active', 'current_recovered', 'current_deceased', 'current_hospitalized', 
                'current_icu', 'current_ventilator', 'active_mean', 'active_min',
                'active_max', 'hospitalized_mean', 'hospitalized_min', 'hospitalized_max', 'icu_mean', 'icu_min', 
                'icu_max', 'deceased_mean', 'deceased_min', 'deceased_max', 'recovered_mean', 'recovered_min', 
                'recovered_max', 'total_mean', 'total_min', 'total_max']
    df_output = pd.DataFrame(columns=columns)

    if df_prediction is None:
        df_prediction = get_forecast(predictions_dict, model=model, best_params=best_params, days=days)

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

def create_decile_csv(predictions_dict: dict, region: str, regionType: str, icu_fraction=0.02):
    print("compiling csv data ..")
    columns = ['forecastRunDate', 'regionType', 'region', 'model_name', 'error_function', 'current_total', 'current_active', 'current_recovered',
               'current_deceased', 'current_hospitalised', 'current_icu', 'current_ventilator', 'predictionDate']
    
    for decile in predictions_dict['m2']['forecasts'].keys():
        columns += [f'active_{decile}',
            f'hospitalised_{decile}',
            f'icu_{decile}',
            f'recovered_{decile}',
            f'deceased_{decile}',
            f'total_{decile}',
            f'error_{decile}',
        ]

    df_output = pd.DataFrame(columns=columns)

    df_true = predictions_dict['m2']['df_district']

    dateseries = predictions_dict['m2']['forecasts'][list(
        predictions_dict['m2']['forecasts'].keys())[0]]['date']
    prediction_daterange = np.union1d(df_true['date'], dateseries)
    no_of_data_points = len(prediction_daterange)
    df_output['predictionDate'] = prediction_daterange

    df_output['forecastRunDate'] = [datetime.datetime.today().date()]*no_of_data_points
    df_output['regionType'] = [regionType]*no_of_data_points
    df_output['region'] = [region]*no_of_data_points
    df_output['model_name'] = ['SEIR']*no_of_data_points
    df_output['error_function'] = ['MAPE']*no_of_data_points
    df_output.set_index('predictionDate', inplace=True)

    for decile in predictions_dict['m2']['forecasts'].keys():
        df_prediction = predictions_dict['m2']['forecasts'][decile]
        df_prediction = df_prediction.set_index('date')
        # df_loss = predictions_dict[decile]['df_loss']
        df_output.loc[df_prediction.index, f'active_{decile}'] = df_prediction['hospitalised']
        df_output.loc[df_prediction.index, f'hospitalised_{decile}'] = df_prediction['hospitalised']
        # df_output.loc[df_prediction.index, f'icu_{decile}'] = icu_fraction*df_prediction['hospitalised']
        df_output.loc[df_prediction.index, f'recovered_{decile}'] = df_prediction['recovered']
        df_output.loc[df_prediction.index, f'deceased_{decile}'] = df_prediction['deceased']
        df_output.loc[df_prediction.index, f'total_{decile}'] = df_prediction['total_infected']
        # df_output.loc[df_prediction.index, f'error_{decile}'] = df_loss.loc[:, 'train'].sum()

    df_true = df_true.set_index('date')
    df_output.loc[df_true.index, 'current_total'] = df_true['total_infected'].to_numpy()
    df_output.loc[df_true.index, 'current_hospitalised'] = df_true['hospitalised'].to_numpy()
    df_output.loc[df_true.index, 'current_deceased'] = df_true['deceased'].to_numpy()
    df_output.loc[df_true.index, 'current_recovered'] = df_true['recovered'].to_numpy()
    
    df_output.reset_index(inplace=True)
    # df_output = df_output[columns]
    return df_output

def create_all_csvs(predictions_dict: dict, district:str='Mumbai', days=30, icu_fraction=0.02):
    """Creates the output for all geographical regions (not just one)

    Arguments:
        predictions_dict {dict} -- The predictions dict for all geographical regions

    Keyword Arguments:
        icu_fraction {float} -- Percentage of active cases that are in the ICU (default: {0.02})

    Returns:
        pd.DataFrame -- output for all geographical regions
    """
    columns = ['forecastRunDate', 'predictionDate', 'regionType', 'region', 'model_name', 'error_function', 'error_value', 'which_forecast',
                'current_total', 'current_active', 'current_recovered', 'current_deceased', 'current_hospitalized', 
                'current_icu', 'current_ventilator', 'active_mean', 'active_min', 'active_max', 
                'hospitalized_mean', 'hospitalized_min', 'hospitalized_max', 'icu_mean', 'icu_min', 'icu_max', 
                'deceased_mean', 'deceased_min', 'deceased_max', 'recovered_mean', 'recovered_min', 'recovered_max', 
                'total_mean', 'total_min', 'total_max']
    df_final = pd.DataFrame(columns=columns)
    for forecast, df_prediction in predictions_dict['m2']['forecasts'].items():
        df_output = create_region_csv(predictions_dict, region=district, regionType='district',
                                      df_prediction=df_prediction, icu_fraction=icu_fraction, days=days)
        df_output['which_forecast'] = forecast
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

def _order_trials_by_loss(m_dict: dict):
    """Orders a set of trials by their corresponding loss value

    Args:
        m_dict (dict): predictions_dict for a particular train_fit

    Returns:
        array, array: Array of params and loss values resp
    """
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

def _get_top_k_trials(m_dict: dict, k=10):
    """Returns Top k trials ordered by loss

    Args:
        m_dict (dict): predictions_dict for a particular train_fit
        k (int, optional): Number of trials. Defaults to 10.

    Returns:
        array, array: array of params and losses resp (of len k each)
    """
    params_array, losses_array = _order_trials_by_loss(m_dict)
    return params_array[:k], losses_array[:k]

def forecast_top_k_trials(predictions_dict: dict, model=SEIRHD, k=10, train_fit='m2', forecast_days=37):
    """Creates forecasts for the top k Bayesian Opt trials (ordered by loss) for a specified number of days

    Args:
        predictions_dict (dict): The dict of predictions for a particular region
        k (int, optional): The number of trials to forecast for. Defaults to 10.
        train_fit (str, optional): Which train fit (m1 or m2). Defaults to 'm2'.
        forecast_days (int, optional): Number of days to forecast for. Defaults to 37.

    Returns:
        array, array, array: array of predictions, losses, and parameters resp
    """
    top_k_params, top_k_losses = _get_top_k_trials(predictions_dict[train_fit], k=k)
    predictions = []
    simulate_till = datetime.datetime.strptime(predictions_dict[train_fit]['data_last_date'], '%Y-%m-%d') + \
        datetime.timedelta(days=forecast_days)
    print("getting forecasts ..")
    for i, params_dict in tqdm(enumerate(top_k_params)):
        predictions.append(get_forecast(predictions_dict, best_params=params_dict, model=model, 
                                        train_fit=train_fit, simulate_till=simulate_till, verbose=False))
    return predictions, top_k_losses, top_k_params


def forecast_all_trials(predictions_dict, model=SEIRHD, train_fit='m2', forecast_days=37):
    """Forecasts all trials in a particular train_fit, in predictions dict

    Args:
        predictions_dict (dict): The dict of predictions for a particular region
        train_fit (str, optional): Which train fit (m1 or m2). Defaults to 'm2'.
        forecast_days (int, optional): How many days to forecast for. Defaults to 37.

    Returns:
        [type]: [description]
    """
    predictions, losses, params = forecast_top_k_trials(
        predictions_dict, 
        k=len(predictions_dict[train_fit]['trials']), 
        model=model,
        train_fit=train_fit,
        forecast_days=forecast_days
    )
    return_dict = {
        'predictions': predictions, 
        'losses': losses, 
        'params': params
    }
    return return_dict

def trials_to_df(trials_processed, column=Columns.active):
    predictions = trials_processed['predictions']
    params = trials_processed['params']
    losses = trials_processed['losses']
    
    cols = ['loss', 'compartment']
    for key in params[0].keys():
        cols.append(key)
    trials = pd.DataFrame(columns=cols)
    for i in range(len(params)):
        to_add = copy.copy(params[i])
        to_add['loss'] = losses[i]
        to_add['compartment'] = column.name
        trials = trials.append(to_add, ignore_index=True)
    pred = pd.DataFrame(columns=predictions[0]['date'])
    for i in range(len(params)):
        pred = pred.append(predictions[i].set_index('date').loc[:, [column.name]].transpose(), ignore_index=True)
    return pd.concat([trials, pred], axis=1)


def scale_up_testing_and_forecast(predictions_dict, which_fit='m2', model=SEIRHD, scenario_on_which_df='best', 
                                  testing_scaling_factor=1.5, time_window_to_scale=14):
    
    df_whatif = scale_up_acc_to_testing(predictions_dict, scenario_on_which_df=scenario_on_which_df, 
                                        testing_scaling_factor=testing_scaling_factor,
                                        time_window_to_scale=time_window_to_scale)

    optimiser = Optimiser()
    extra_params = optimiser.init_default_params(df_whatif, N=1e7, initialisation='intermediate', 
                                                 train_period=time_window_to_scale)
    best_params = copy.copy(predictions_dict[which_fit]['best_params'])
    del best_params['T_inf']
    del best_params['E_hosp_ratio']
    del best_params['I_hosp_ratio']
    default_params = {**extra_params, **best_params}

    total_days = (df_whatif.iloc[-1, :]['date'] - default_params['starting_date']).days
    variable_param_ranges = {
        'T_inf': (0.01, 10),
        'E_hosp_ratio': (0, 2),
        'I_hosp_ratio': (0, 1)
    }
    variable_param_ranges = get_variable_param_ranges(variable_param_ranges=variable_param_ranges)
    best, trials = optimiser.bayes_opt(df_whatif, default_params, variable_param_ranges, model=model,
                                       total_days=total_days, method='mape', num_evals=500, 
                                       loss_indices=[-time_window_to_scale, None], 
                                       which_compartments=['total_infected'])

    df_unscaled_forecast = predictions_dict[which_fit]['forecasts'][scenario_on_which_df]

    df_prediction = optimiser.solve(best, default_params, 
                                    predictions_dict[which_fit]['df_train'], 
                                    end_date=df_unscaled_forecast.iloc[-1, :]['date'], 
                                    model=model)
    return df_prediction


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


def predict_r0_multipliers(region_dict, params_dict, days, model=SEIRHD,
                           multipliers=[0.9, 1, 1.1, 1.25], lockdown_removal_date='2020-06-01'):
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
            model=model,
            best_params=new_params,
            lockdown_removal_date=lockdown_removal_date,
            days=days)    
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
