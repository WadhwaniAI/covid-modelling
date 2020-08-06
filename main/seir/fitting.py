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
from data.processing import granular

from models.seir.seir_testing import SEIR_Testing
from main.seir.optimiser import Optimiser
from utils.loss import Loss_Calculator
from utils.enums import Columns
from utils.smooth_jump import smooth_big_jump, smooth_big_jump_stratified
from viz import plot_smoothing, plot_fit

def get_variable_param_ranges(variable_param_ranges=None, initialisation='intermediate', as_str=False, 
                              mode='hyperopt', searchspace_len=21):
    """Returns the ranges for the variable params in the search space

    Keyword Arguments:
        initialisation {str} -- The method of initialisation (default: {'intermediate'})
        as_str {bool} -- If true, the parameters are not returned as a hyperopt object, but as a dict in 
        string form (default: {False})

    Returns:
        dict -- dict of ranges of variable params
    """
    if variable_param_ranges == None:
        variable_param_ranges = {
            'lockdown_R0': (1, 1.3),
            'T_inc': (4, 5),
            'T_inf': (3, 4),
            'T_recov_severe': (5, 60),
            'T_recov_fatal': (0, 40),
            'P_fatal': (0, 0.3),
            'E_hosp_ratio': (0, 2),
            'I_hosp_ratio': (0, 1)
        }
    if initialisation != 'intermediate':
        del variable_param_ranges['E_hosp_ratio']
        del variable_param_ranges['I_hosp_ratio']
    if as_str:
        return str(variable_param_ranges)

    if mode == 'hyperopt':
        for key in variable_param_ranges.keys():
            variable_param_ranges[key] = hp.uniform(
                key, variable_param_ranges[key][0], variable_param_ranges[key][1])

    if mode == 'gridsearch':
        for key in variable_param_ranges.keys():
            variable_param_ranges[key] = np.linspace(
                variable_param_ranges[key][0], variable_param_ranges[key][1], searchspace_len)

    return variable_param_ranges
   
def train_val_split(df_district, train_rollingmean=False, val_rollingmean=False, val_size=5, rolling_window=5,
                    which_columns=['total_infected', 'hospitalised', 'recovered', 'deceased'], continuous_ra=True):
    """Creates train val split on dataframe

    # TODO : Add support for creating train val test split

    Arguments:
        df_district {pd.DataFrame} -- The observed dataframe

    Keyword Arguments:
        train_rollingmean {bool} -- If true, apply rolling mean on train (default: {False})
        val_rollingmean {bool} -- If true, apply rolling mean on val (default: {False})
        val_size {int} -- Size of val set (default: {5})
        rolling_window {int} -- Size of rolling window. The rolling window is centered (default: {5})
        continuous_ra {bool} -- If true, end values of train use actuals from val set for rolling mean

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame -- train dataset, val dataset, concatenation of rolling average dfs
    """
    print("splitting data ..")
    df_true_fitting = copy.copy(df_district)
    # Perform rolling average on all columns with numeric datatype
    df_true_fitting = df_true_fitting.infer_objects()
    if which_columns == None:
        which_columns = df_true_fitting.select_dtypes(include='number').columns
    for column in which_columns:
        if column in df_true_fitting.columns:
            df_true_fitting[column] = df_true_fitting[column].rolling(
                rolling_window, center=True).mean()

    # Since the rolling average method is center, we need an offset variable where the ends of the series will
    # use the true observations instead (as rolling averages for those offset days don't exist)
    offset_window = rolling_window // 2

    df_true_fitting.dropna(axis=0, how='any', subset=which_columns, inplace=True)
    df_true_fitting.reset_index(inplace=True, drop=True)

    if train_rollingmean:
        if val_size == 0:
            df_train = pd.concat(
                [df_true_fitting, df_district.iloc[-(val_size+offset_window):, :]], ignore_index=True)
            return df_train, None
        else:
            if continuous_ra:
                df_train = df_true_fitting.iloc[:-(val_size-offset_window), :]
            else:
                df_train = pd.concat(
                    [df_true_fitting.iloc[:-val_size, :], df_district.iloc[-(val_size+offset_window):-val_size, :]],
                    ignore_index=True)
    else:
        if val_size == 0:
            return df_district, None
        else:
            df_train = df_district.iloc[:-val_size, :]

    if val_rollingmean:
        df_val = pd.concat([df_true_fitting.iloc[-(val_size-offset_window):, :],
                            df_district.iloc[-offset_window:, :]], ignore_index=True)
    else:
        df_val = df_district.iloc[-val_size:, :]
    df_val.reset_index(inplace=True, drop=True)

    return df_train, df_val
    

def get_regional_data(state, district, data_from_tracker, data_format, filename, which_compartments,
                      granular_data=False, smooth_jump=False, t_recov=14, return_extra=False):
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
    if granular_data:
        df_not_strat = get_data(
            state=state, district=district, filename=filename, disable_tracker=True)
        df_district = granular.get_data(filename=filename)
    else:
        if data_from_tracker:
            df_district = get_data(state=state, district=district, use_dataframe='data_all')
        else:
            df_district = get_data(state=state, district=district, disable_tracker=True, filename=filename, 
                                   data_format=data_format)
    
    smoothing_plot = None
    orig_df_district = copy.copy(df_district)

    if smooth_jump:
        if granular_data:
            df_district, description = smooth_big_jump_stratified(
                df_district, df_not_strat, smooth_stratified_additionally=True)
        else:
            df_district, description = smooth_big_jump(df_district, data_from_tracker=data_from_tracker)

        smoothing_plot = plot_smoothing(orig_df_district, df_district, state, district,
                                        which_compartments=which_compartments, description=f'Smoothing')
    
    df_district['daily_cases'] = df_district['total_infected'].diff()

    if return_extra:
        extra = {
            'smoothing_description': description,
            'smoothing_plot': smoothing_plot,
            'df_district_unsmoothed': orig_df_district
        }
        print(extra['smoothing_description'])
        return df_district, extra 
    return df_district 


def data_setup(df_district, val_period, continuous_ra=True):
    """Helper function for single_fitting_cycle which sets up the data including doing the train val split

    Arguments:
        df_district {pd.DataFrame} -- True observations from districts_daily/custom file/athena DB
        val_period {int} -- Length of val period

    Returns:
        dict(pd.DataFrame) -- Dict of pd.DataFrame objects
    """
    # Get train val split 
    df_train, df_val = train_val_split(
        df_district, train_rollingmean=True, val_rollingmean=True, val_size=val_period, rolling_window=7,
        continuous_ra=continuous_ra)
    df_train_nora, df_val_nora = train_val_split(
        df_district, train_rollingmean=False, val_rollingmean=False, val_size=val_period)

    observed_dataframes = {}
    for name in ['df_district', 'df_train', 'df_val', 'df_train_nora', 'df_val_nora']:
        observed_dataframes[name] = eval(name)
    return observed_dataframes


def run_cycle(state, district, observed_dataframes, model=SEIR_Testing, variable_param_ranges=None, 
              default_params=None, train_period=7, data_from_tracker=True,
              which_compartments=['hospitalised', 'total_infected', 'recovered', 'deceased'], 
              num_evals=1500, N=1e7, initialisation='starting', back_offset=0):
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

    df_district, df_train, df_val, df_train_nora, df_val_nora = [
        observed_dataframes.get(k) for k in observed_dataframes.keys()]

    # Initialise Optimiser
    optimiser = Optimiser()
    # Get the fixed params
    if initialisation == 'starting':
        observed_values = df_district.iloc[0, :]
        start_date = df_district.iloc[0, :]['date']
        std_default_params = optimiser.init_default_params(df_train, N=N, observed_values=observed_values,
                                                           start_date=start_date, initialisation=initialisation)
    elif initialisation == 'intermediate':
        std_default_params = optimiser.init_default_params(df_train, N=N, initialisation=initialisation,
                                                           train_period=train_period)
    if default_params is not None:
        default_params = {**std_default_params, **default_params}
    else:
        default_params = std_default_params
    # Get/create searchspace of variable params
    if variable_param_ranges == None:
        variable_param_ranges = get_variable_param_ranges(initialisation=initialisation)

    if back_offset == 0:
        loss_indices = [-train_period, None]
    else:
        loss_indices = [-(train_period+back_offset), -back_offset]
    # Perform Bayesian Optimisation
    total_days = (df_train.iloc[-1, :]['date'] - default_params['starting_date']).days
    best_params, trials = optimiser.bayes_opt(df_train, default_params, variable_param_ranges, model=model, 
                                              num_evals=num_evals, loss_indices=loss_indices, method='mape',
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

    pointwise_train_loss, pointwise_val_loss = lc.create_pointwise_loss_dataframe_region(df_train_nora, df_val_nora, df_prediction, train_period,
                                               which_compartments=which_compartments)

    fit_plot = plot_fit(df_prediction, df_train, df_val, df_district, train_period, state, district,
                        which_compartments=which_compartments)

    results_dict = {}
    results_dict['plots'] = {}
    results_dict['plots']['fit'] = fit_plot
    data_last_date = df_district.iloc[-1]['date'].strftime("%Y-%m-%d")
    variable_param_ranges = get_variable_param_ranges(initialisation=initialisation, as_str=True)
    for name in ['data_from_tracker', 'best_params', 'default_params', 'variable_param_ranges', 'optimiser', 
                 'df_prediction', 'df_district', 'df_train', 'df_val', 'df_loss', 'pointwise_train_loss',
                 'pointwise_val_loss', 'plot_fit', 'trials', 'data_last_date']:

        results_dict[name] = eval(name)

    return results_dict


def single_fitting_cycle(state, district, model=SEIR_Testing, variable_param_ranges=None, default_params=None, #Main 
                         data_from_tracker=True, granular_data=False, filename=None, data_format='new', #Data
                         train_period=7, val_period=7, num_evals=1500, N=1e7, initialisation='starting', back_offset=0,  #Misc
                         which_compartments=['hospitalised', 'total_infected'], #Compartments
                         smooth_jump=False, continuous_ra=True): #Smoothing

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
    run_params['model'] = model.__name__
    run_params['model_class'] = model
    
    print('Performing {} fit ..'.format('m2' if val_period == 0 else 'm1'))

    # Get data
    df_district, extra = get_regional_data(
        state, district, data_from_tracker, data_format, filename, 
        which_compartments=which_compartments, granular_data=granular_data,
        smooth_jump=smooth_jump, return_extra=True
    )
    smoothing_plot = extra['smoothing_plot']
    orig_df_district = extra['df_district_unsmoothed']

    # Process the data to get rolling averages and other stuff
    observed_dataframes = data_setup(df_district, val_period, continuous_ra=continuous_ra)

    print('train\n', observed_dataframes['df_train'].tail())
    print('val\n', observed_dataframes['df_val'])
    
    predictions_dict = run_cycle(
        state, district, observed_dataframes, 
        model=model, variable_param_ranges=variable_param_ranges, default_params=default_params,
        data_from_tracker=data_from_tracker, train_period=train_period, 
        which_compartments=which_compartments, N=N, back_offset=back_offset,
        num_evals=num_evals, initialisation=initialisation
    )

    if smoothing_plot != None:
        predictions_dict['plots']['smoothing'] = smoothing_plot
        predictions_dict['smoothing_description'] = extra['smoothing_description']
    predictions_dict['df_district_unsmoothed'] = orig_df_district

    # record parameters for reproducibility
    predictions_dict['run_params'] = run_params

    return predictions_dict
