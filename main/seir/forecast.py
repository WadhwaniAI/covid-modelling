import datetime
import os

import numpy as np
import pandas as pd


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


def predict_r0_multipliers(region_dict, params_dict, days, model,
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
        params_dict = {**new_params, **region_dict['default_params']}
        solver = model(**params_dict)
        predictions_mul_dict[mul]['df_prediction'] = solver.predict(
            total_days=days) 
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
