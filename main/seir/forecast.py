from main.seir.optimiser import Optimiser
import os
from hyperopt import Trials
import numpy as np
import pandas as pd

from tqdm import tqdm
import datetime
import copy

from models.seir import SEIRHD

def get_forecast(predictions_dict: dict, forecast_days: int = 37, train_end_date=None, model=SEIRHD,
                 best_params=None):
    """Returns the forecasts for a given set of params of a particular geographical area

    Arguments:
        predictions_dict {dict} -- [description]

    Keyword Arguments:
        train_end_date {[type]} -- [description] (default: {None})
        best_params {[type]} -- [description] (default: {None})

    Returns:
        [type] -- [description]
    """
    if train_end_date is None:
        simulate_till = predictions_dict['df_district'].iloc[-1]['date'] + \
            datetime.timedelta(days=forecast_days)
    else:
        simulate_till = train_end_date + datetime.timedelta(days=forecast_days)
        simulate_till = datetime.datetime.combine(simulate_till, datetime.datetime.min.time())
    if best_params == None:
        best_params = predictions_dict['best_params']

    default_params = copy.copy(predictions_dict['default_params'])
    
    op = Optimiser()
    df_prediction = op.solve({**best_params, **default_params}, model=model, 
                             end_date=simulate_till)

    return df_prediction

def create_all_trials_csv(predictions_dict: dict):
    df_all = pd.DataFrame(columns=predictions_dict['trials_processed']['predictions'][0].columns)
    for i, df_prediction in enumerate(predictions_dict['trials_processed']['predictions']):
        df_prediction['loss'] = predictions_dict['trials_processed']['losses'][i]
        df_all = pd.concat([df_all, df_prediction])

    forecast_columns = [x for x in df_all.columns if not x[0].isupper()]

    return df_all[forecast_columns]

def create_decile_csv_new(predictions_dict: dict):
    """Nayana's implementation of the CSV format that P&P consume for the presentations

    Args:
        predictions_dict (dict): Dict of all predictions

    Returns:
        pd.DataFrame: Dataframe in the format that Keshav wants
    """
    forecast_columns = [x for x in predictions_dict['forecasts']['best'].columns if not x[0].isupper()]
    forecast_columns = [x for x in forecast_columns if x != 'date']
    column_mapping = {k:k for k in forecast_columns}

    df_percentiles_list = []
    percentile_labels = []

    for decile, df_prediction in predictions_dict['forecasts'].items():
        if decile == 'best':
            continue
        percentile_labels.append(" ".join([str(decile), "Percentile"]))
        percentiles = [decile] * len(forecast_columns)
        percentile_columns = ["".join([col, str(decile)]) for col in column_mapping.values()]
        index_arrays = [percentiles, percentile_columns, column_mapping.values()]
        layered_index = pd.MultiIndex.from_arrays(index_arrays)
        df = pd.DataFrame(columns=layered_index)
        for column in forecast_columns:
            df.loc[:, (decile, "".join([column_mapping[column], str(decile)]), column_mapping[column])] = df_prediction[column]
        df_percentiles_list.append(df)
    df_output = pd.concat(df_percentiles_list, keys=percentile_labels, axis=1)
    df_output.insert(0, 'Date', df_prediction['date'])
    
    return df_output

def create_decile_csv(predictions_dict: dict, region: str, regionType: str):
    print("compiling csv data ..")
    columns = ['forecastRunDate', 'regionType', 'region', 'model_name', 'error_function', 'predictionDate',
               'current_total', 'current_active', 'current_recovered', 'current_deceased']
    
    forecast_columns = [x for x in predictions_dict['forecasts']['best'].columns if not x[0].isupper()]
    forecast_columns = [x for x in forecast_columns if x != 'date']

    for decile in predictions_dict['forecasts'].keys():
        columns += [f'{x}_{decile}' for x in forecast_columns]

    df_output = pd.DataFrame(columns=columns)

    df_true = predictions_dict['df_district']

    dateseries = predictions_dict['forecasts'][list(
        predictions_dict['forecasts'].keys())[0]]['date']
    prediction_daterange = np.union1d(df_true['date'], dateseries)
    no_of_data_points = len(prediction_daterange)
    df_output['predictionDate'] = prediction_daterange

    df_output['forecastRunDate'] = [datetime.datetime.strptime(
        predictions_dict['fitting_date'], '%Y-%m-%d')]*no_of_data_points
    df_output['regionType'] = [regionType]*no_of_data_points
    df_output['region'] = [region]*no_of_data_points
    df_output['model_name'] = [predictions_dict['run_params']['model']]*no_of_data_points
    df_output['error_function'] = ['MAPE']*no_of_data_points
    df_output.set_index('predictionDate', inplace=True)

    for decile, df_prediction in predictions_dict['forecasts'].items():
        df_prediction = df_prediction.set_index('date')
        for column in forecast_columns:
            df_output.loc[df_prediction.index, f'{column}_{decile}'] = df_prediction[column]

    df_true = df_true.set_index('date')
    df_output.loc[df_true.index, 'current_total'] = df_true['total_infected'].to_numpy()
    df_output.loc[df_true.index, 'current_active'] = df_true['hospitalised'].to_numpy()
    df_output.loc[df_true.index, 'current_deceased'] = df_true['deceased'].to_numpy()
    df_output.loc[df_true.index, 'current_recovered'] = df_true['recovered'].to_numpy()
    
    df_output.reset_index(inplace=True)
    df_output.columns = [x.replace('hospitalised', 'active') for x in df_output.columns]
    df_output.columns = [x.replace('total_infected', 'total') for x in df_output.columns]
    return df_output


def _order_trials_by_loss_hp(trials_obj: Trials, sort_trials: bool = True):
    """Orders a set of trials by their corresponding loss value

    Args:
        m_dict (dict): predictions_dict

    Returns:
        array, array: Array of params and loss values resp
    """
    params_array = []
    for trial in trials_obj:
        params_dict = copy.copy(trial['misc']['vals'])
        for key in params_dict.keys():
            params_dict[key] = params_dict[key][0]
        params_array.append(params_dict)
    params_array = np.array(params_array)
    losses_array = np.array([trial['result']['loss'] for trial in trials_obj])

    if sort_trials:
        least_losses_indices = np.argsort(losses_array)
        losses_array = losses_array[least_losses_indices]
        params_array = params_array[least_losses_indices]
    return params_array, losses_array


def _get_top_k_trials(trials_obj: Trials, k=10):
    """Returns Top k trials ordered by loss

    Args:
        m_dict (dict): predictions_dict
        k (int, optional): Number of trials. Defaults to 10.

    Returns:
        array, array: array of params and losses resp (of len k each)
    """
    params_array, losses_array = _order_trials_by_loss_hp(trials_obj)
    return params_array[:k], losses_array[:k]


def forecast_top_k_trials(predictions_dict: dict, model=SEIRHD, k=10, train_end_date=None, 
                          forecast_days=37):
    """Creates forecasts for the top k Bayesian Opt trials (ordered by loss) for a specified number of days

    Args:
        predictions_dict (dict): The dict of predictions for a particular region
        k (int, optional): The number of trials to forecast for. Defaults to 10.
        forecast_days (int, optional): Number of days to forecast for. Defaults to 37.

    Returns:
        array, array, array: array of predictions, losses, and parameters resp
    """
    top_k_params, top_k_losses = _get_top_k_trials(predictions_dict, k=k)
    predictions = []
    print("getting forecasts ..")
    for i, params_dict in tqdm(enumerate(top_k_params)):
        predictions.append(get_forecast(predictions_dict, best_params=params_dict, model=model, 
                                        train_end_date=train_end_date, 
                                        forecast_days=forecast_days, verbose=False))
    return predictions, top_k_losses, top_k_params


def forecast_all_trials(predictions_dict, model=SEIRHD, train_end_date=None, forecast_days=37):
    """Forecasts all trials in a particular, in predictions dict

    Args:
        predictions_dict (dict): The dict of predictions for a particular region
        forecast_days (int, optional): How many days to forecast for. Defaults to 37.

    Returns:
        [type]: [description]
    """
    predictions, losses, params = forecast_top_k_trials(
        predictions_dict, 
        k=len(predictions_dict['trials']), 
        model=model,
        train_end_date=train_end_date,
        forecast_days=forecast_days
    )
    return_dict = {
        'predictions': predictions, 
        'losses': losses, 
        'params': params
    }
    return return_dict

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
                           multipliers=[0.9, 1, 1.1, 1.25]):
    """
    Function to predict what-if scenarios with different post-lockdown R0s

    Args:
        region_dict (dict): region_dict as returned by main.seir.main.single_fitting_cycle
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
            model=model,
            best_params=new_params,
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
    columns_for_csv = ['date', 'total', 'active', 'recovered', 'deceased']
    for (mul, val) in predictions_mul_dict.items():
        df_prediction = val['df_prediction']
        path = f'../../misc/reports/{folder}/what-ifs/'
        if not os.path.exists(path):
            os.makedirs(path)
        df_prediction[columns_for_csv].to_csv(os.path.join(path, f'what-if-{mul}.csv'))
    pd.DataFrame({key: val['params'] for key, val in predictions_mul_dict.items()}) \
        .to_csv(f'../../misc/reports/{folder}/what-ifs/what-ifs-params.csv')
