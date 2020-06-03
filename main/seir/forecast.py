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
import datetime
from joblib import Parallel, delayed
import copy

from data.dataloader import get_jhu_data, get_covid19india_api_data
from data.processing import get_data, get_district_time_series

from models.seir.seir_testing import SEIR_Testing
from main.seir.optimiser import Optimiser
from main.seir.losses import Loss_Calculator

from utils.enums import Columns

def get_forecast(predictions_dict: dict, simulate_till=None, train_fit='m2', best_params=None):
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
    print("getting forecasts ..")
    if simulate_till == None:
        simulate_till = datetime.datetime.today() + datetime.timedelta(days=37)
    if best_params == None:
        best_params = predictions_dict[train_fit]['best_params']
    df_prediction = predictions_dict[train_fit]['optimiser'].solve(best_params,
                                                                   predictions_dict[train_fit]['default_params'],
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


def preprocess_for_error_plot(df_prediction : pd.DataFrame, df_loss : pd.DataFrame, 
                              which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered']):
    df_temp = copy.copy(df_prediction)
    df_temp.loc[:, which_compartments] = df_prediction.loc[:, which_compartments]*(1 - 0.01*df_loss['val'])
    df_prediction = pd.concat([df_prediction, df_temp], ignore_index=True)
    df_temp = copy.copy(df_prediction)
    df_temp.loc[:, which_compartments] = df_prediction.loc[:, which_compartments]*(1 + 0.01*df_loss['val'])
    df_prediction = pd.concat([df_prediction, df_temp], ignore_index=True)
    return df_prediction

def plot_forecast(predictions_dict : dict, region : tuple, both_forecasts=False, log_scale=False, filename=None, 
                  which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'], 
                  fileformat='eps', error_bars=False):
    """Function for plotting forecasts

    Arguments:
        predictions_dict {dict} -- Dict of predictions for a particular district 
        region {tuple} -- Region Name eg : ('Maharashtra', 'Mumbai')

    Keyword Argument
        which_compartments {list} -- Which compartments to plot (default: {['hospitalised', 'total_infected', 'deceased', 'recovered']})
        both_forecasts {bool} -- If true, plot both forecasts (default: {False})
        log_scale {bool} -- If true, y is in log scale (default: {False})
        filename {str} -- If given, the plot is saved here (default: {None})
        fileformat {str} -- The format in which the plot will be saved (default: {'eps'})
        error_bars {bool} -- If true, error bars will be plotted (default: {False})

    Returns:
        ax -- Matplotlib ax figure
    """
    df_prediction = get_forecast(predictions_dict)
    # df_prediction.loc[:, which_compartments] = df_prediction[]
    if both_forecasts:
        df_prediction_m1 = get_forecast(predictions_dict, train_fit='m1')
    df_true = predictions_dict['m1']['df_district']
    
    if error_bars:
        df_prediction = preprocess_for_error_plot(df_prediction, predictions_dict['m1']['df_loss'], 
                                                  which_compartments)
        if both_forecasts:
            df_prediction_m1 = preprocess_for_error_plot(df_prediction_m1, predictions_dict['m1']['df_loss'], 
                                                        which_compartments)

    fig, ax = plt.subplots(figsize=(12, 12))

    if 'total_infected' in which_compartments:
        ax.plot(df_true['date'], df_true['total_infected'],
                '-o', color='C0', label='Confirmed Cases (Observed)')
        sns.lineplot(x="date", y="total_infected", data=df_prediction,
                     ls='-', color='C0', label='Confirmed Cases (M2 Forecast)')
        if both_forecasts:
            sns.lineplot(x="date", y="total_infected", data=df_prediction_m1,
                         ls='--', color='C0', label='Confirmed Cases (M1 Forecast)')
            # ax.plot(df_prediction_m1['date'], df_prediction_m1['total_infected'],
            #         '--', color='C0', label='Confirmed Cases (M1 Forecast)')
    if 'hospitalised' in which_compartments:
        ax.plot(df_true['date'], df_true['hospitalised'],
                '-o', color='orange', label='Active Cases (Observed)')
        sns.lineplot(x="date", y="hospitalised", data=df_prediction,
                     ls='-', color='orange', label='Active Cases (M2 Forecast)')
        if both_forecasts:
            sns.lineplot(x="date", y="hospitalised", data=df_prediction_m1,
                         ls='--', color='orange', label='Active Cases (M1 Forecast)')
            # ax.plot(df_prediction_m1['date'], df_prediction_m1['hospitalised'],
            #         '--', color='orange', label='Active Cases (M1 Forecast)')
    if 'recovered' in which_compartments:
        ax.plot(df_true['date'], df_true['recovered'],
                '-o', color='green', label='Recovered Cases (Observed)')
        sns.lineplot(x="date", y="recovered", data=df_prediction,
                     ls='-', color='green', label='Recovered Cases (M2 Forecast)')
        if both_forecasts:
            sns.lineplot(x="date", y="recovered", data=df_prediction_m1,
                         ls='--', color='green', label='Recovered Cases (M1 Forecast)')
            # ax.plot(df_prediction_m1['date'], df_prediction_m1['recovered'],
            #         '--', color='green', label='Recovered Cases (M1 Forecast)')
    if 'deceased' in which_compartments:
        ax.plot(df_true['date'], df_true['deceased'],
                '-o', color='red', label='Deceased Cases (Observed)')
        sns.lineplot(x="date", y="deceased", data=df_prediction,
                     ls='-', color='red', label='Deceased Cases (M2 Forecast)')
        if both_forecasts:
            sns.lineplot(x="date", y="deceased", data=df_prediction_m1,
                         ls='--', color='red', label='Deceased Cases (M1 Forecast)')
            # ax.plot(df_prediction_m1['date'], df_prediction_m1['deceased'],
            #         '--', color='red', label='Deceased Cases (M1 Forecast)')
    
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.ylabel('No of People', fontsize=16)
    if log_scale:
        plt.yscale('log')
    plt.xlabel('Time', fontsize=16)
    plt.xticks(rotation=45,horizontalalignment='right')
    plt.legend()
    plt.title('Forecast - ({} {})'.format(region[0], region[1]), fontsize=16)
    plt.grid()
    if filename != None:
        plt.savefig(filename, format=fileformat)

    return ax

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

def forecast_k(predictions_dict: dict, k=10, train_fit='m2'):
    top_k_losses, top_k_params = top_k_trials(predictions_dict[train_fit], k=k)
    predictions = [get_forecast(
        predictions_dict, best_params=params_dict, train_fit=train_fit) for params_dict in top_k_params]
    return predictions, top_k_losses

def plot_trials(predictions_dict, k=10, train_fit='m2', which_compartments=[Columns.active]):
    predictions, top_k_losses = forecast_k(predictions_dict, k=k, train_fit=train_fit)
    df_true = predictions_dict[train_fit]['df_district']
    fig, ax = plt.subplots(figsize=(12, 12))
    texts = []
    for compartment in which_compartments:
        ax.plot(df_true[Columns.date.name], df_true[compartment.name],
                '-o', color='C0', label=f'{compartment.label} (Observed)')
        for i, df_prediction in enumerate(predictions):
            loss_value = np.around(top_k_losses[i], 2)
            sns.lineplot(x=Columns.date.name, y=compartment.name, data=df_prediction,
                        ls='-', label=f'{compartment.label} ({loss_value})')
            texts.append(plt.text(
                x=df_prediction[Columns.date.name].iloc[-1] + datetime.timedelta(
                    days=np.random.binomial(1, 0.5)), 
                y=df_prediction[compartment.name].iloc[-1], s=loss_value))
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 10)
    from adjustText import adjust_text
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.ylabel('No of People', fontsize=16)
    plt.yscale('log')
    plt.xlabel('Time', fontsize=16)
    plt.legend()
    plt.title('Forecast - ({} {})'.format(predictions_dict['state'], predictions_dict['dist']), fontsize=16)
    # plt.grid()
    return ax