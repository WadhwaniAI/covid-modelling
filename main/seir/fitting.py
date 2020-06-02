import os
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

from data.processing import get_all_district_data

from models.seir.seir_testing import SEIR_Testing
from main.seir.optimiser import Optimiser
from main.seir.losses import Loss_Calculator


now = str(datetime.now())

def get_variable_param_ranges(initialisation='intermediate', as_str=False):
    variable_param_ranges = {
        'lockdown_R0': (1, 1.5),
        'T_inc': (4, 5),
        'T_inf': (3, 4),
        'T_recov_severe': (5, 60),
        'P_severe': (0.3, 0.99),
        'P_fatal': (0, 0.3)
    }
    if initialisation == 'intermediate':
        extra_params = {
            'E_hosp_ratio': (0, 2),
            'I_hosp_ratio': (0, 1)
        }
        variable_param_ranges.update(extra_params)
    if as_str:
        return str(variable_param_ranges)

    for key in variable_param_ranges.keys():
        variable_param_ranges[key] = hp.uniform(
            key, variable_param_ranges[key][0], variable_param_ranges[key][1])

    return variable_param_ranges
    

def train_val_split(df_district, train_rollingmean=False, val_rollingmean=False, val_size=5,
                    which_columns=['hospitalised', 'total_infected', 'deceased', 'recovered']):
    print("splitting data ..")
    df_true_fitting = copy.copy(df_district)
    for column in which_columns:
        df_true_fitting[column] = df_true_fitting[column].rolling(
            5, center=True).mean()

    df_true_fitting = df_true_fitting[np.logical_not(
        df_true_fitting['total_infected'].isna())]
    df_true_fitting.reset_index(inplace=True, drop=True)

    if train_rollingmean:
        if val_size == 0:
            df_train = pd.concat(
                [df_true_fitting, df_district.iloc[-(val_size+2):, :]], ignore_index=True)
            return df_train, None, df_true_fitting
        else:
            df_train = df_true_fitting.iloc[:-(val_size-2), :]
    else:
        if val_size == 0:
            return df_district, None, df_true_fitting
        else:
            df_train = df_district.iloc[:-val_size, :]

    if val_rollingmean:
        df_val = pd.concat([df_true_fitting.iloc[-(val_size-2):, :],
                            df_district.iloc[-2:, :]], ignore_index=True)
    else:
        df_val = df_district.iloc[-val_size:, :]
    df_val.reset_index(inplace=True, drop=True)
    return df_train, df_val, df_true_fitting
    
def data_setup(df_district, df_district_raw_data, pre_lockdown, 
                    train_on_val, val_period):
    
    # Get train val split
    if pre_lockdown:
        df_train, df_val, df_true_fitting = train_val_split(
            df_district_raw_data, train_rollingmean=False, val_rollingmean=False, val_size=0)
    else:
        if train_on_val:
            df_train, df_val, df_true_fitting = train_val_split(
                df_district, train_rollingmean=True, val_rollingmean=True, val_size=0)
            df_train_nora, df_val_nora, df_true_fitting = train_val_split(
                df_district, train_rollingmean=False, val_rollingmean=False, val_size=0)
        else:
            df_train, df_val, df_true_fitting = train_val_split(
                df_district, train_rollingmean=True, val_rollingmean=True, val_size=val_period)
            df_train_nora, df_val_nora, df_true_fitting = train_val_split(
                df_district, train_rollingmean=False, val_rollingmean=False, val_size=val_period)
    return df_district, df_district_raw_data, df_train, df_val, df_true_fitting, df_train_nora, df_val_nora

def run_cycle(state, district, df_district, df_district_raw_data, df_train, df_val, df_train_nora, df_val_nora, data_from_tracker,
                        train_period=7, train_on_val=False, num_evals=1500, N=1e7, 
                        which_compartments=['hospitalised', 'total_infected'], initialisation='starting'):
    if district is None:
        district = ''
    
    # Initialise Optimiser
    optimiser = Optimiser()
    # Get the fixed params
    if initialisation == 'starting':
        observed_values = df_district_raw_data.iloc[0, :]
        start_date = df_district_raw_data.iloc[0, :]['date']
        default_params = optimiser.init_default_params(df_train, observed_values=observed_values,
                                                       start_date=start_date, initialisation=initialisation, N=N)
    elif initialisation == 'intermediate':
        default_params = optimiser.init_default_params(df_train, N=N, initialisation=initialisation, 
                                                       train_period=train_period)

    # Get/create searchspace of variable params
    variable_param_ranges = get_variable_param_ranges(initialisation=initialisation)

    # Perform Bayesian Optimisation
    total_days = (df_train.iloc[-1, :]['date'] - default_params['starting_date']).days + 1
    best_params, trials = optimiser.bayes_opt(df_train, default_params, variable_param_ranges, method='mape', 
                                              num_evals=num_evals, loss_indices=[-train_period, None],
                                              total_days=total_days, which_compartments=which_compartments, 
                                              initialisation=initialisation)

    print('best parameters\n', best_params)

    if train_on_val:
        df_prediction = optimiser.solve(best_params, default_params, df_train, end_date=df_train.iloc[-1, :]['date'], 
                                        initialisation=initialisation, loss_indices=[-train_period, None])
    else:
        df_prediction = optimiser.solve(best_params, default_params, df_train, end_date=df_val.iloc[-1, :]['date'],
                                        initialisation=initialisation, loss_indices=[-train_period, None])
    
    df_loss = calculate_loss(df_train_nora, df_val_nora, df_prediction, train_period,
                             train_on_val, which_compartments=which_compartments)

    
    ax = create_plots(df_prediction, df_train, df_val, df_train_nora, df_val_nora, train_period, state, district,
                      which_compartments=which_compartments)

    results_dict = {}
    data_last_date = df_district.iloc[-1]['date'].strftime("%Y-%m-%d")
    variable_param_ranges = get_variable_param_ranges(initialisation=initialisation, as_str=True)
    for name in ['data_from_tracker', 'best_params', 'default_params', 'variable_param_ranges', 'optimiser', 
                 'df_prediction', 'df_district', 'df_train', 'df_val', 'df_loss', 'ax', 'trials', 'data_last_date']:
        results_dict[name] = eval(name)

    return results_dict

def single_fitting_cycle(dataframes, state, district, train_period=7, val_period=7, train_on_val=False, num_evals=1500,
                         data_from_tracker=True, filename=None, data_format='new', pre_lockdown=False, N=1e7, 
                         which_compartments=['hospitalised', 'total_infected'], initialisation='starting'):
    # Get date
    print('fitting to data with "train_on_val" set to {} ..'.format(train_on_val))
    df_district, df_district_raw_data = get_all_district_data(dataframes, state, district, 
        data_from_tracker, data_format, filename)

    df_district, df_district_raw_data, df_train, df_val, df_true_fitting, df_train_nora, df_val_nora = data_setup(
        df_district, df_district_raw_data, pre_lockdown, train_on_val, val_period
    )

    print('train\n', df_train.tail())
    print('val\n', df_val)
    
    return run_cycle(
        state, district, df_district, df_district_raw_data, df_train, df_val, df_train_nora, df_val_nora, data_from_tracker,
        train_period=train_period, train_on_val=train_on_val, num_evals=num_evals, N=N, 
        which_compartments=which_compartments, initialisation=initialisation
    )

def calculate_loss(df_train, df_val, df_prediction, train_period,
                   train_on_val, which_compartments=['hospitalised', 'total_infected']):
    loss_calculator = Loss_Calculator()
    df_loss = pd.DataFrame(columns=['train', 'val'], index=which_compartments)

    df_temp = df_prediction.loc[df_prediction['date'].isin(df_train['date']), [
        'date', 'hospitalised', 'total_infected', 'deceased', 'recovered']]
    df_temp.reset_index(inplace=True, drop=True)
    df_train.reset_index(inplace=True, drop=True)
    for compartment in df_loss.index:
        df_loss.loc[compartment, 'train'] = loss_calculator._calc_mape(
            np.array(df_train[compartment].iloc[-train_period:]), np.array(df_temp[compartment].iloc[-train_period:]))

    if isinstance(df_val, pd.DataFrame):
        df_temp = df_prediction.loc[df_prediction['date'].isin(df_val['date']), [
            'date', 'hospitalised', 'total_infected', 'deceased', 'recovered']]
        df_temp.reset_index(inplace=True, drop=True)
        df_val.reset_index(inplace=True, drop=True)
        for compartment in df_loss.index:
            df_loss.loc[compartment, 'val'] = loss_calculator._calc_mape(
                np.array(df_val[compartment]), np.array(df_temp[compartment]))
    else:
        del df_loss['val']
    return df_loss

def create_plots(df_prediction, df_train, df_val, df_train_nora, df_val_nora, train_period, state, district, 
                 which_compartments=['hospitalised', 'total_infected'], description=''):
    # Create plots
    fig, ax = plt.subplots(figsize=(12, 12))
    if isinstance(df_val, pd.DataFrame):
        df_true_plotting_rolling = pd.concat([df_train, df_val], ignore_index=True)
        df_true_plotting = pd.concat([df_train_nora, df_val_nora], ignore_index=True)
    else:
        df_true_plotting_rolling = df_train
        df_true_plotting = df_train_nora
    
    df_predicted_plotting = df_prediction.loc[df_prediction['date'].isin(
        df_true_plotting['date']), ['date', 'hospitalised', 'total_infected', 'deceased', 'recovered']]
    
    if 'total_infected' in which_compartments:
        ax.plot(df_true_plotting['date'], df_true_plotting['total_infected'],
                '-o', color='C0', label='Confirmed Cases (Observed)')
        ax.plot(df_true_plotting_rolling['date'], df_true_plotting_rolling['total_infected'],
                '-', color='C0', label='Confirmed Cases (Obs RA)')
        ax.plot(df_predicted_plotting['date'], df_predicted_plotting['total_infected'],
                '-.', color='C0', label='Confirmed Cases (Predicted)')
    if 'hospitalised' in which_compartments:
        ax.plot(df_true_plotting['date'], df_true_plotting['hospitalised'],
                '-o', color='orange', label='Active Cases (Observed)')
        ax.plot(df_true_plotting_rolling['date'], df_true_plotting_rolling['hospitalised'],
                '-', color='orange', label='Active Cases (Obs RA)')
        ax.plot(df_predicted_plotting['date'], df_predicted_plotting['hospitalised'],
                '-.', color='orange', label='Active Cases (Predicted)')
    if 'recovered' in which_compartments:
        ax.plot(df_true_plotting['date'], df_true_plotting['recovered'],
                '-o', color='green', label='Recovered Cases (Observed)')
        ax.plot(df_true_plotting_rolling['date'], df_true_plotting_rolling['recovered'],
                '-', color='green', label='Recovered Cases (Obs RA)')
        ax.plot(df_predicted_plotting['date'], df_predicted_plotting['recovered'],
                '-.', color='green', label='Recovered Cases (Predicted)')
    if 'deceased' in which_compartments:
        ax.plot(df_true_plotting['date'], df_true_plotting['deceased'],
                '-o', color='red', label='Deceased Cases (Observed)')
        ax.plot(df_true_plotting_rolling['date'], df_true_plotting_rolling['deceased'],
                '-', color='red', label='Deceased Cases (Obs RA)')
        ax.plot(df_predicted_plotting['date'], df_predicted_plotting['deceased'],
                '-.', color='red', label='Deceased Cases (Predicted)')
    
    ax.plot([df_train.iloc[-train_period, :]['date'], df_train.iloc[-train_period, :]['date']],
            [min(df_train['deceased']), max(df_train['total_infected'])], '--', color='brown', label='Train starts')
    if isinstance(df_val, pd.DataFrame):
        ax.plot([df_val.iloc[0, :]['date'], df_val.iloc[0, :]['date']],
                [min(df_val['deceased']), max(df_val['total_infected'])], '--', color='black', label='Val starts')

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.ylabel('No of People')
    plt.xlabel('Time')
    plt.legend()
    plt.title('{} - ({} {})'.format(description, state, district))
    plt.grid()

    return ax
