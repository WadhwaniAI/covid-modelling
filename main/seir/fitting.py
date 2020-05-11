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


now = str(datetime.now())

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
            df_train = pd.concat([df_true_fitting.iloc[:-val_size, :], df_district.iloc[-(val_size+2):-val_size, :]],
                                 ignore_index=True)
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


def single_fitting_cycle(dataframes, state, district, train_period=7, val_period=7, train_on_val=False,
                         data_from_tracker=True, filename=None, pre_lockdown=False, 
                         which_compartments=['hospitalised', 'total_infected']):

    print('fitting to data with "train_on_val" set to {} ..'.format(train_on_val))

    if data_from_tracker:
        df_district = get_data(dataframes, state=state, district=district, use_dataframe='districts_daily')
    else:
        df_district = get_data(dataframes, state, district, disable_tracker=True, filename=filename)
    
    df_district_raw_data = get_data(dataframes, state=state, district=district, use_dataframe='raw_data')
    df_district_raw_data = df_district_raw_data[df_district_raw_data['date'] <= '2020-03-25']

    if district is None:
        district = ''

    # Get train val split
    if pre_lockdown:
        df_train, df_val, df_true_fitting = train_val_split(
            df_district_raw_data, train_rollingmean=False, val_rollingmean=False, val_size=0)
    else:
        if train_on_val:
            df_train, df_val, df_true_fitting = train_val_split(
                df_district, train_rollingmean=False, val_rollingmean=False, val_size=0)
        else:
            df_train, df_val, df_true_fitting = train_val_split(
                df_district, train_rollingmean=False, val_rollingmean=False, val_size=val_period)

    print('train\n', df_train.tail())
    print('val\n', df_val)

    # Initialise Optimiser
    optimiser = Optimiser()
    # Get the fixed params
    init_infected = max(df_district_raw_data.iloc[0, :]['total_infected'], 1)
    start_date = df_district_raw_data.iloc[0, :]['date']
    default_params = optimiser.init_default_params(df_train, N=9.43e6, init_infected=init_infected, 
                                                   start_date=start_date)

    # Create searchspace of variable params
    variable_param_ranges = {
        'R0': hp.uniform('R0', 1.6, 5),
        'T_inc': hp.uniform('T_inc', 4, 5),
        'T_inf': hp.uniform('T_inf', 3, 4),
        'T_recov_severe': hp.uniform('T_recov_severe', 9, 20),
        'P_severe': hp.uniform('P_severe', 0.3, 0.9),
        'P_fatal': hp.uniform('P_fatal', 0, 0.3),
        'intervention_amount': hp.uniform('intervention_amount', 0, 1)
    }

    # Perform Bayesian Optimisation
    total_days = (df_train.iloc[-1, :]['date'] - default_params['starting_date']).days + 1
    best_params, trials = optimiser.bayes_opt(df_train, default_params, variable_param_ranges, method='mape', 
                                              num_evals=1500, loss_indices=[-train_period, None],
                                              total_days=total_days, which_compartments=which_compartments)

    print('best parameters\n', best_params)

    if train_on_val:
        df_prediction = optimiser.solve(best_params, default_params, df_train, end_date=df_train.iloc[-1, :]['date'])
    else:
        df_prediction = optimiser.solve(best_params, default_params, df_train, end_date=df_val.iloc[-1, :]['date'])
    
    df_loss = calculate_loss(df_train, df_val, df_prediction, train_period,
                             train_on_val, which_compartments=which_compartments)

    
    ax = create_plots(df_prediction, df_train, df_val, train_period, state, district,
                      which_compartments=which_compartments)

    results_dict = {}
    for name in ['best_params', 'default_params', 'optimiser', 'df_prediction', 'df_district', 'df_train', \
        'df_val', 'df_loss', 'ax']:
        results_dict[name] = eval(name)

    return results_dict


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

def create_plots(df_prediction, df_train, df_val, train_period, state, district, 
                 which_compartments=['hospitalised', 'total_infected'], description=''):
    # Create plots
    fig, ax = plt.subplots(figsize=(12, 12))
    if isinstance(df_val, pd.DataFrame):
        df_true_plotting = pd.concat([df_train, df_val], ignore_index=True)
    else:
        df_true_plotting = df_train
    
    df_predicted_plotting = df_prediction.loc[df_prediction['date'].isin(
        df_true_plotting['date']), ['date', 'hospitalised', 'total_infected', 'deceased', 'recovered']]

    if 'total_infected' in which_compartments:
        ax.plot(df_true_plotting['date'], df_true_plotting['total_infected'],
                '-', color='C0', label='Confirmed Cases (Observed)')
        ax.plot(df_true_plotting['date'], df_predicted_plotting['total_infected'],
                '-.', color='C0', label='Confirmed Cases (Predicted)')
    if 'hospitalised' in which_compartments:
        ax.plot(df_true_plotting['date'], df_true_plotting['hospitalised'],
                '-', color='orange', label='Active Cases (Observed)')
        ax.plot(df_true_plotting['date'], df_predicted_plotting['hospitalised'],
                '-.', color='orange', label='Active Cases (Predicted)')
    if 'recovered' in which_compartments:
        ax.plot(df_true_plotting['date'], df_true_plotting['recovered'],
                '-', color='green', label='Recovered Cases (Observed)')
        ax.plot(df_true_plotting['date'], df_predicted_plotting['recovered'],
                '-.', color='green', label='Recovered Cases (Predicted)')
    if 'deceased' in which_compartments:
        ax.plot(df_true_plotting['date'], df_true_plotting['deceased'],
                '-', color='red', label='Deceased Cases (Observed)')
        ax.plot(df_true_plotting['date'], df_predicted_plotting['deceased'],
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
