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
import datetime
from joblib import Parallel, delayed
import copy

from data.processing import get_data

from models.seir.seir_testing import SEIR_Testing
from main.seir.optimiser import Optimiser
from main.seir.losses import Loss_Calculator


now = str(datetime.datetime.now())

def get_variable_param_ranges(initialisation='intermediate', as_str=False):
    """Returns the ranges for the variable params in the search space

    Keyword Arguments:
        initialisation {str} -- The method of initialisation (default: {'intermediate'})
        as_str {bool} -- If true, the parameters are not returned as a hyperopt object, but as a dict in 
        string form (default: {False})

    Returns:
        dict -- dict of ranges of variable params
    """
    variable_param_ranges = {
        'lockdown_R0': (1, 1.5),
        'T_inc': (4, 5),
        'T_inf': (3, 4),
        'T_recov_severe': (5, 60),
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
    

def train_val_split(df_district, train_rollingmean=False, val_rollingmean=False, val_size=5, rolling_window=5, 
                    which_columns=['hospitalised', 'total_infected', 'deceased', 'recovered']):
    """Creates train val split on dataframe

    # TODO : Add support for creating train val test split

    Arguments:
        df_district {pd.DataFrame} -- The observed dataframe

    Keyword Arguments:
        train_rollingmean {bool} -- If true, apply rolling mean on train (default: {False})
        val_rollingmean {bool} -- If true, apply rolling mean on val (default: {False})
        val_size {int} -- Size of val set (default: {5})
        rolling_window {int} -- Size of rolling window. The rolling window is centered (default: {5})
        which_columns {list} -- Which columnns to do the rolling average over (default: {['hospitalised', 'total_infected', 'deceased', 'recovered']})

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame -- train dataset, val dataset, concatenation of rolling average dfs
    """
    print("splitting data ..")
    df_true_fitting = copy.copy(df_district)
    for column in which_columns:
        df_true_fitting[column] = df_true_fitting[column].rolling(
            rolling_window, center=True).mean()

    # Since the rolling average method is center, we need an offset variable where the ends of the series will
    # use the true observations instead (as rolling averages for those offset days don't exist)
    offset_window = rolling_window // 2

    df_true_fitting = df_true_fitting[np.logical_not(
        df_true_fitting['total_infected'].isna())]
    df_true_fitting.reset_index(inplace=True, drop=True)

    if train_rollingmean:
        if val_size == 0:
            df_train = pd.concat(
                [df_true_fitting, df_district.iloc[-(val_size+offset_window):, :]], ignore_index=True)
            return df_train, None, df_true_fitting
        else:
            df_train = df_true_fitting.iloc[:-(val_size-offset_window), :]
    else:
        if val_size == 0:
            return df_district, None, df_true_fitting
        else:
            df_train = df_district.iloc[:-val_size, :]

    if val_rollingmean:
        df_val = pd.concat([df_true_fitting.iloc[-(val_size-offset_window):, :],
                            df_district.iloc[-offset_window:, :]], ignore_index=True)
    else:
        df_val = df_district.iloc[-val_size:, :]
    df_val.reset_index(inplace=True, drop=True)
    return df_train, df_val, df_true_fitting
    
def get_regional_data(dataframes, state, district, data_from_tracker, data_format, filename, smooth_jump=False,
                      smoothing_length=28, smoothing_method='linear'):
    """Helper function for single_fitting_cycle where data from different sources (given input) is imported

    Arguments:
        dataframes {dict(pd.Dataframe)} -- dict of dataframes from get_covid19india_api_data() 
        state {str} -- State name in title case
        district {str} -- District name in title case
        data_from_tracker {bool} -- Whether data is from tracker or not
        data_format {str} -- If using filename, what is the filename format
        filename {str} -- Name of the filename to read file from

    Returns:
        pd.DataFrame, pd.DataFrame -- data from main source, and data from raw_data in covid19india
    """
    if data_from_tracker:
        df_district = get_data(dataframes, state=state, district=district, use_dataframe='districts_daily')
    else:
        df_district = get_data(state=state, district=district, disable_tracker=True, filename=filename, 
                               data_format=data_format)
    
    df_district_raw_data = get_data(dataframes, state=state, district=district, use_dataframe='raw_data')

    if smooth_jump:
        df_district = smooth_big_jump(
            df_district, smoothing_length=smoothing_length, 
            method=smoothing_method, data_from_tracker=data_from_tracker)

    return df_district, df_district_raw_data

def smooth_big_jump(df_district, smoothing_length, data_from_tracker, t_recov=14, method='linear', ):
    if data_from_tracker:
        d1, d2 = '2020-05-29', '2020-05-30'
    else:
        d1, d2 = '2020-05-28', '2020-05-29'
    df_district['date'] = pd.to_datetime(df_district['date'])
    df_district = df_district.set_index('date')
    big_jump = df_district.loc[d2, 'recovered'] - df_district.loc[d1, 'recovered']
    print(big_jump)
    if method == 'linear':
        for i, day_number in enumerate(range(smoothing_length-2, -1, -1)):
            date = datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.timedelta(days=day_number)
            offset = np.random.binomial(1, (big_jump%smoothing_length)/smoothing_length)
            df_district.loc[date, 'recovered'] += ((i+1)*big_jump)//smoothing_length + offset
            df_district.loc[date, 'hospitalised'] -= ((i+1)*big_jump)//smoothing_length + offset

    elif method == 'weighted':
        newcases = df_district['total_infected'].shift(t_recov) - df_district['total_infected'].shift(t_recov + 1)
        valid_idx = newcases.first_valid_index()
        window_start = datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.timedelta(days=smoothing_length - 1)
        newcases = newcases.loc[max(valid_idx, window_start):d1]
        truncated = df_district.loc[max(valid_idx, window_start):d1, :]
        print('len truncated', len(truncated))
        invpercent = newcases.sum()/newcases
        for i, day_number in enumerate(range(smoothing_length-2, -1, -1)):
            date = datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.timedelta(days=day_number)
            offset = np.random.binomial(1, (big_jump%invpercent.loc[date])/invpercent.loc[date])
            truncated.loc[date:, 'recovered'] += (big_jump // invpercent.loc[date]) + offset
            truncated.loc[date:, 'hospitalised'] -= (big_jump // invpercent.loc[date]) + offset
        df_district.loc[truncated.index, 'recovered'] = truncated['recovered'].astype('int64')
        df_district.loc[truncated.index, 'hospitalised'] = truncated['hospitalised'].astype('int64')

    assert((df_district['total_infected'] == df_district['hospitalised'] + df_district['deceased'] + df_district['recovered']).all())
    return df_district.reset_index()


def data_setup(df_district, df_district_raw_data, val_period, which_columns=['hospitalised', 'total_infected', 'deceased', 'recovered']):
    """Helper function for single_fitting_cycle which sets up the data including doing the train val split

    Arguments:
        df_district {pd.DataFrame} -- True observations from districts_daily/custom file/athena DB
        df_district_raw_data {pd.DataFrame} -- True observations from raw_data
        val_period {int} -- Length of val period

    Returns:
        dict(pd.DataFrame) -- Dict of pd.DataFrame objects
    """
    # Get train val split
    df_train, df_val, df_true_fitting = train_val_split(
        df_district, train_rollingmean=True, val_rollingmean=True, val_size=val_period, which_columns=which_columns)
    df_train_nora, df_val_nora, df_true_fitting = train_val_split(
        df_district, train_rollingmean=False, val_rollingmean=False, val_size=val_period, which_columns=which_columns)

    observed_dataframes = {}
    for name in ['df_district', 'df_district_raw_data', 'df_train', 'df_val', 'df_train_nora', 'df_val_nora']:
        observed_dataframes[name] = eval(name)
    return observed_dataframes


def run_cycle(state, district, observed_dataframes, model=SEIR_Testing, data_from_tracker=True, train_period=7, 
              which_compartments=['hospitalised', 'total_infected', 'recovered', 'deceased'], num_evals=1500, N=1e7, 
              initialisation='starting'):
    """Helper function for single_fitting_cycle where the fitting actually takes place

    Arguments:
        state {str} -- state name in title case
        district {str} -- district name in title case
        observed_dataframes {dict(pd.DataFrame)} -- Dict of all observed dataframes

    Keyword Arguments:
        model {class} -- The epi model class we're using to perform optimisation (default: {SEIR_Testing})
        data_from_tracker {bool} -- If true, data is from covid19india API (default: {True})
        train_period {int} -- Length of training period (default: {7})
        which_compartments {list} -- Whci compartments to apply loss over 
        (default: {['hospitalised', 'total_infected', 'recovered', 'deceased']})
        num_evals {int} -- Number of evaluations for hyperopt (default: {1500})
        N {float} -- Population of area (default: {1e7})
        initialisation {str} -- Method of initialisation (default: {'starting'})

    Returns:
        dict -- Dict of all predictions
    """
    if district is None:
        district = ''

    df_district, df_district_raw_data, df_train, df_val, df_train_nora, df_val_nora = [
        observed_dataframes.get(k) for k in observed_dataframes.keys()]

    # Initialise Optimiser
    optimiser = Optimiser()
    # Get the fixed params
    if initialisation == 'starting':
        observed_values = df_district_raw_data.iloc[0, :]
        start_date = df_district_raw_data.iloc[0, :]['date']
        default_params = optimiser.init_default_params(df_train, N=N, observed_values=observed_values,
                                                       start_date=start_date, initialisation=initialisation)
    elif initialisation == 'intermediate':
        default_params = optimiser.init_default_params(df_train, N=N, initialisation=initialisation, 
                                                       train_period=train_period)

    # Get/create searchspace of variable params
    variable_param_ranges = get_variable_param_ranges(initialisation=initialisation)

    # Perform Bayesian Optimisation
    total_days = (df_train.iloc[-1, :]['date'] - default_params['starting_date']).days + 1
    best_params, trials = optimiser.bayes_opt(df_train, default_params, variable_param_ranges, model=model, 
                                              num_evals=num_evals, loss_indices=[-train_period, None], method='mape',
                                              total_days=total_days, which_compartments=which_compartments)

    print('best parameters\n', best_params)

    if not isinstance(df_val, pd.DataFrame):
        df_prediction = optimiser.solve(best_params, default_params, df_train, end_date=df_train.iloc[-1, :]['date']) 
    else:
        df_prediction = optimiser.solve(best_params, default_params, df_train, end_date=df_val.iloc[-1, :]['date'])
    
    df_loss = calculate_loss(df_train_nora, df_val_nora, df_prediction, train_period, 
                             which_compartments=which_compartments)

    
    ax = create_plots(df_prediction, df_train, df_val, df_train_nora, df_val_nora, train_period, state, district,
                      which_compartments=['hospitalised', 'total_infected', 'recovered', 'deceased'])

    results_dict = {}
    data_last_date = df_district.iloc[-1]['date'].strftime("%Y-%m-%d")
    variable_param_ranges = get_variable_param_ranges(initialisation=initialisation, as_str=True)
    for name in ['data_from_tracker', 'best_params', 'default_params', 'variable_param_ranges', 'optimiser', 
                 'df_prediction', 'df_district', 'df_train', 'df_val', 'df_loss', 'ax', 'trials', 'data_last_date']:
        results_dict[name] = eval(name)

    return results_dict

def single_fitting_cycle(dataframes, state, district, model=SEIR_Testing, train_period=7, val_period=7, 
                         data_from_tracker=True, filename=None, data_format='new', N=1e7, num_evals=1500,
                         which_compartments=['hospitalised', 'total_infected'], initialisation='starting', 
                         smooth_jump=False, smoothing_length=28, smoothing_method='linear'):
    """Main function which user runs for running an entire fitting cycle for a particular district

    Arguments:
        dataframes {dict(pd.DataFrame)} -- Dict of dataframes returned by the get_covid19india_api_data function
        state {str} -- State Name
        district {str} -- District Name (in title case)

    Keyword Arguments:
        model_class {class} -- The epi model class to be used for modelling (default: {SEIR_Testing})
        train_period {int} -- The training period (default: {7})
        val_period {int} -- The validation period (default: {7})
        num_evals {int} -- Number of evaluations of Bayesian Optimsation (default: {1500})
        data_from_tracker {bool} -- If False, data from tracker is not used (default: {True})
        filename {str} -- If None, Athena database is used. Otherwise, data in filename is read (default: {None})
        data_format {str} -- The format type of the filename user is providing ('old'/'new') (default: {'new'})
        N {float} -- The population of the geographical region (default: {1e7})
        which_compartments {list} -- Which compartments to fit on (default: {['hospitalised', 'total_infected']})
        initialisation {str} -- The method of intitalisation (default: {'starting'})

    Returns:
        dict -- dict of everything related to prediction
    """
    print('Performing {} fit ..'.format('m2' if val_period == 0 else 'm1'))

    # Get data
    df_district, df_district_raw_data = get_regional_data(dataframes, state, district, data_from_tracker, data_format, 
                                                          filename, smooth_jump=smooth_jump, smoothing_method=smoothing_method,
                                                          smoothing_length=smoothing_length)

    # Process the data to get rolling averages and other stuff
    observed_dataframes = data_setup(
        df_district, df_district_raw_data, 
        val_period, which_columns=which_compartments)

    print('train\n', observed_dataframes['df_train'].tail())
    print('val\n', observed_dataframes['df_val'])
    
    predictions_dict = run_cycle(
        state, district, observed_dataframes, 
        data_from_tracker=data_from_tracker, 
        model=model, train_period=train_period, 
        which_compartments=which_compartments, N=N,
        num_evals=num_evals, initialisation=initialisation
    )

    # record parameters for reproducability
    predictions_dict['run_params'] = {
        'state': state,
        'district': district,
        'model': model.__name__,
        'train_period': train_period,
        'val_period': val_period,
        'data_from_tracker': data_from_tracker,
        'filename': filename,
        'data_format': data_format,
        'N': N,
        'num_evals': num_evals,
        'which_compartments': which_compartments,
        'initialisation': initialisation,
        'smooth_jump': smooth_jump,
        'smoothing_length': smoothing_length,
        'smoothing_method': smoothing_method,
    }

    return predictions_dict

def calculate_loss(df_train, df_val, df_prediction, train_period, which_compartments=['hospitalised', 'total_infected']):
    """Helper function for calculating loss in training pipeline

    Arguments:
        df_train {pd.DataFrame} -- Train dataset
        df_val {pd.DataFrame} -- Val dataset
        df_prediction {pd.DataFrame} -- Model Prediction
        train_period {int} -- Length of training Period

    Keyword Arguments:
        which_compartments {list} -- List of buckets to calculate loss on (default: {['hospitalised', 'total_infected']})

    Returns:
        pd.DataFrame -- A dataframe of train loss values and val (if val exists too)
    """
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
    """Helper function for creating plots for the training pipeline

    Arguments:
        df_prediction {pd.DataFrame} -- The prediction dataframe outputted by the model
        df_train {pd.DataFrame} -- The train dataset (with rolling average)
        df_val {pd.DataFrame} -- The val dataset (with rolling average)
        df_train_nora {pd.DataFrame} -- The train dataset (with no rolling average)
        df_val_nora {pd.DataFrame} -- The val dataset (with no rolling average)
        train_period {int} -- Length of train period
        state {str} -- Name of state
        district {str} -- Name of district

    Keyword Arguments:
        which_compartments {list} -- Which buckets to plot (default: {['hospitalised', 'total_infected']})
        description {str} -- Additional description for the plots (if any) (default: {''})

    Returns:
        ax -- Matplotlib ax object
    """
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
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.legend()
    plt.title('{} - ({} {})'.format(description, state, district))
    plt.grid()

    return ax
