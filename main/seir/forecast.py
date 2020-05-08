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

from collections import OrderedDict, defaultdict
import itertools
from functools import partial
from datetime import datetime
from joblib import Parallel, delayed
import copy

from data.dataloader import get_jhu_data, get_covid19india_api_data
from data.processing import get_data, get_district_time_series

from models.seir.seir_testing import SEIR_Testing
from main.seir.optimiser import Optimiser
from main.seir.losses import Loss_Calculator

def get_forecast(predictions_dict: dict, simulate_till=None):
    print("getting forecasts ..")
    if simulate_till == None:
        simulate_till = datetime.datetime.today() + datetime.timedelta(days=30)
    else:
        simulate_till = 
    best_params = copy.copy(predictions_dict['m2']['best'])
    df_prediction = predictions_dict['m2']['optimiser'].solve(predictions_dict['m2']['best_params'],
                                                              predictions_dict['m2']['default_params'],
                                                              predictions_dict['m2']['df_train'], 
                                                              end_date=simulate_till)

    return df_prediction


def create_csv_data(predictions_dict: dict):
    print("compiling csv data ..")
    simulate_till = datetime.strptime(end_date, '%Y-%m-%d')
    dfs = defaultdict()
    for region in forecasts:
        state, district = region
        
        columns = ['forecastRunDate', 'regionType', 'region', 'model_name', 'error_function', 'error_value', 'current_total', 'current_active', 'current_recovered', 
               'current_deceased', 'current_hosptialized', 'current_icu', 'current_ventilator', 'predictionDate', 'active_mean', 'active_min', 
               'active_max', 'hospitalized_mean', 'hospitalized_min', 'hospitalized_max', 'icu_mean', 'icu_min', 'icu_max', 'deceased_mean', 
               'deceased_min', 'deceased_max', 'recovered_mean', 'recovered_min', 'recovered_max', 'total_mean', 'total_min', 'total_max']

        df_output = pd.DataFrame(columns = columns)
        
        city = predictions_dict[region].copy()
        df_prediction = get_forecast(predictions_dict)
        prediction_daterange = df_prediction['date']
        no_of_predictions = len(prediction_daterange)
        
        df_output['predictionDate'] = prediction_daterange
        df_output['forecastRunDate'] = [datetime.today().date()]*no_of_predictions
        
        df_output['regionType'] = ['city']*no_of_predictions
        
        df_output['model_name'] = ['SEIR']*no_of_predictions
        df_output['error_function'] = ['MAPE']*no_of_predictions
            
        error = [predictions_dict['m1']['df_loss'].loc['total_infected', 'val']]
        df_output['error_value'] = [error[0]]*no_of_predictions

        pred_hospitalisations = df_prediction['hospitalised']
        df_output['active_mean'] = pred_hospitalisations
        df_output['active_min'] = (1 - 0.01*error[0])*pred_hospitalisations
        df_output['active_max'] = (1 + 0.01*error[0])*pred_hospitalisations
        
        df_output['hospitalized_mean'] = pred_hospitalisations
        df_output['hospitalized_min'] = (1 - 0.01*error[0])*pred_hospitalisations
        df_output['hospitalized_max'] = (1 - 0.01*error[0])*pred_hospitalisations
        
        df_output['icu_mean'] = 0.02*pred_hospitalisations
        df_output['icu_min'] = (1 - 0.01*error[0])*0.02*pred_hospitalisations
        df_output['icu_max'] = (1 - 0.01*error[0])*0.02*pred_hospitalisations
        
        pred_recoveries = df_prediction['recovered']
        df_output['recovered_mean'] = pred_recoveries
        df_output['recovered_min'] = (1 - 0.01*error[0])*pred_recoveries
        df_output['recovered_max'] = (1 - 0.01*error[0])*pred_recoveries
        
        pred_fatalities = df_prediction['deceased']
        df_output['deceased_mean'] = pred_fatalities
        df_output['deceased_min'] = (1 - 0.01*error[0])*pred_fatalities
        df_output['deceased_max'] = (1 - 0.01*error[0])*pred_fatalities
        
        pred_total_cases = df_prediction['total_infected']
        df_output['total_mean'] = pred_total_cases
        df_output['total_min'] = (1 - 0.01*error[0])*pred_total_cases
        df_output['total_max'] = (1 - 0.01*error[0])*pred_total_cases
        
        if state == 'Delhi':
            district = 'Delhi'
        df_output['region'] = [district]*no_of_predictions
        
        df_output.set_index('predictionDate', inplace=True)
        df_district = predictions_dict_val_train[region]['df_district']
        df_output.loc[df_output.index.isin(df_district['date']), 'current_total'] = df_district['total_infected'].iloc[2:].to_numpy()
        df_output.reset_index(inplace=True)
        df_output = df_output[columns]
        
        dfs[region] = df_output

    return dfs

def write_csv(dfs: dict):
