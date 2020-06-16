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
from utils.loss import Loss_Calculator
from utils.enums import Columns
from utils.smooth_jump import smooth_big_jump
from viz import plot_smoothing, plot_fit

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
                      smoothing_length=28, smoothing_method='uniform', t_recov=14, return_extra=False):
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
    ax = None
    orig_df_district = copy.copy(df_district)

    if smooth_jump:
        df_district = smooth_big_jump(
            df_district, smoothing_length=smoothing_length, 
            method=smoothing_method, data_from_tracker=data_from_tracker, t_recov=t_recov)
        ax = plot_smoothing(orig_df_district, df_district, state, district, description=f'Smoothing: {smoothing_method}')

    if return_extra:
        extra = {
            'ax': ax,
            'df_district_unsmoothed': orig_df_district
        }
        return df_district, df_district_raw_data, extra 
    return df_district, df_district_raw_data 

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


def run_cycle(state, district, observed_dataframes, model=SEIR_Testing, variable_param_ranges=None, train_period=7,
              data_from_tracker=True, which_compartments=['hospitalised', 'total_infected', 'recovered', 'deceased'], 
              num_evals=1500, N=1e7, initialisation='starting'):
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
    if variable_param_ranges == None:
        variable_param_ranges = get_variable_param_ranges(initialisation=initialisation)

    # Perform Bayesian Optimisation
    total_days = (df_train.iloc[-1, :]['date'] - default_params['starting_date']).days
    best_params, trials = optimiser.bayes_opt(df_train, default_params, variable_param_ranges, model=model, 
                                              num_evals=num_evals, loss_indices=[-train_period, None], method='mape',
                                              total_days=total_days, which_compartments=which_compartments)

    print('best parameters\n', best_params)

    if not isinstance(df_val, pd.DataFrame):
        df_prediction = optimiser.solve(best_params, default_params, df_train, end_date=df_train.iloc[-1, :]['date'],
                                        model=model)
    else:
        df_prediction = optimiser.solve(best_params, default_params, df_train, end_date=df_val.iloc[-1, :]['date'], 
                                        model=model)
    
    lc = Loss_Calculator()
    df_loss = lc.create_loss_dataframe_region(df_train_nora, df_val_nora, df_prediction, train_period, 
                                              which_compartments=which_compartments)

    ax = plot_fit(df_prediction, df_train, df_val, df_train_nora, df_val_nora, train_period, state, district,
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
                         smooth_jump=False, smoothing_length=28, smoothing_method='uniform'):
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
    # record parameters for reproducability
    run_params = locals()
    del run_params['dataframes']
    run_params['model'] = model.__name__
    
    print('Performing {} fit ..'.format('m2' if val_period == 0 else 'm1'))

    # Get data
    df_district, df_district_raw_data, extra = get_regional_data(dataframes, state, district, data_from_tracker, data_format,
                                                                         filename, smooth_jump=smooth_jump, smoothing_method=smoothing_method,
                                                                         smoothing_length=smoothing_length, return_extra=True)
    smoothed_plot = extra['ax']
    orig_df_district = extra['df_district_unsmoothed']

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

    if smoothed_plot != None:
        predictions_dict['smoothing_plot'] = smoothed_plot
    predictions_dict['df_district_unsmoothed'] = orig_df_district

    # record parameters for reproducability
    predictions_dict['run_params'] = run_params

    return predictions_dict
