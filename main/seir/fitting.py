import os
import json
import numpy as np
import pandas as pd

from collections import OrderedDict, defaultdict
import datetime
import copy
import importlib
from tabulate import tabulate

from data.processing.processing import get_data, train_val_split
from data.processing import granular

import models.seir
from main.seir.optimiser import Optimiser
from utils.fitting.loss import Loss_Calculator
from utils.generic.enums import Columns
from utils.smooth_jump import smooth_big_jump, smooth_big_jump_stratified
from viz import plot_smoothing, plot_fit


def data_setup(state, district, data_from_tracker, filename, val_period, loss_compartments,
               data_format=None, granular_data=False, smooth_jump=False, **kwargs):
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
        df_not_strat = get_data(state=state, district=district, filename=filename, disable_tracker=True)
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
                                        which_compartments=loss_compartments, description=f'Smoothing')
    
    df_district['daily_cases'] = df_district['total'].diff()
    df_district.dropna(axis=0, how='any', inplace=True)
    df_district.reset_index(drop=True, inplace=True)

    smoothing = {}
    if smooth_jump:
        smoothing = {
            'smoothing_description': description,
            'smoothing_plot': smoothing_plot,
            'df_district_unsmoothed': orig_df_district
        }
        print(smoothing['smoothing_description'])
     
    df_train, df_val = train_val_split(
        df_district, train_rollingmean=True, val_rollingmean=True, val_size=val_period, rolling_window=7)
    df_train_nora, df_val_nora = train_val_split(
        df_district, train_rollingmean=False, val_rollingmean=False, val_size=val_period)

    observed_dataframes = {}
    for name in ['df_district', 'df_train', 'df_val', 'df_train_nora', 'df_val_nora']:
        observed_dataframes[name] = eval(name)
    return observed_dataframes, smoothing


def run_cycle(observed_dataframes, data, model, variable_param_ranges, default_params, fitting_method,
              fitting_method_params, split, loss):
    """Helper function for single_fitting_cycle where the fitting actually takes place

    Arguments:
        state {str} -- state name in title case
        district {str} -- district name in title case
        observed_dataframes {dict(pd.DataFrame)} -- Dict of all observed dataframes

    Keyword Arguments:
        model {class} -- The epi model class we're using to perform optimisation (default: {SEIRHD})
        data_from_tracker {bool} -- If true, data is from covid19india API (default: {True})
        train_period {int} -- Length of training period (default: {7})
        loss_compartments {list} -- Whci compartments to apply loss over 
        (default: {['active', 'total', 'recovered', 'deceased']})
        num_evals {int} -- Number of evaluations for hyperopt (default: {1500})
        N {float} -- Population of area (default: {1e7})

    Returns:
        dict -- Dict of all predictions
    """

    df_district, df_train, df_val, df_train_nora, df_val_nora = [
        observed_dataframes.get(k) for k in observed_dataframes.keys()]

    # Initialise Optimiser
    optimiser = Optimiser()
    # Get the fixed params
    default_params = optimiser.init_default_params(df_train, default_params, train_period=split['train_period'])
    # Get/create searchspace of variable paramms
    loss_indices = [-(split['train_period']), None]
    loss['loss_indices'] = loss_indices
    # Perform Bayesian Optimisation
    variable_param_ranges = optimiser.format_variable_param_ranges(variable_param_ranges, fitting_method)
    args = {'df_train': df_train, 'default_params': default_params, 'variable_param_ranges':variable_param_ranges, 
            'model':model, 'fitting_method': fitting_method, **fitting_method_params, **split, **loss}
    best_params, trials = getattr(optimiser, fitting_method)(**args)
    print('best parameters\n', best_params)

    df_prediction = optimiser.solve({**best_params, **default_params}, 
                                    end_date=df_district.iloc[-1, :]['date'], 
                                    model=model)
    
    lc = Loss_Calculator()
    df_loss = lc.create_loss_dataframe_region(df_train_nora, df_val_nora, df_prediction, split['train_period'], 
                                              which_compartments=loss['loss_compartments'])

    fit_plot = plot_fit(df_prediction, df_train, df_val, df_district, split['train_period'], 
                        data['state'], data['district'], which_compartments=loss['loss_compartments'])

    results_dict = {}
    results_dict['plots'] = {}
    results_dict['plots']['fit'] = fit_plot
    data_last_date = df_district.iloc[-1]['date'].strftime("%Y-%m-%d")
    for name in ['best_params', 'default_params', 'variable_param_ranges', 'optimiser', 
                 'df_prediction', 'df_district', 'df_train', 'df_val', 'df_loss', 'trials', 'data_last_date']:
        results_dict[name] = eval(name)

    return results_dict


def single_fitting_cycle(data, model, variable_param_ranges, default_params, fitting_method, 
                         fitting_method_params, split, loss):
    """Main function which user runs for running an entire fitting cycle for a particular district

    Arguments:
        dataframes {dict(pd.DataFrame)} -- Dict of dataframes returned by the get_covid19india_api_data function
        state {str} -- State Name
        district {str} -- District Name (in title case)

    Keyword Arguments:
        model_class {class} -- The epi model class to be used for modelling (default: {SEIRHD})
        train_period {int} -- The training period (default: {7})
        val_period {int} -- The validation period (default: {7})
        num_evals {int} -- Number of evaluations of Bayesian Optimsation (default: {1500})
        data_from_tracker {bool} -- If False, data from tracker is not used (default: {True})
        filename {str} -- If None, Athena database is used. Otherwise, data in filename is read (default: {None})
        data_format {str} -- The format type of the filename user is providing ('old'/'new') (default: {'new'})
        N {float} -- The population of the geographical region (default: {1e7})
        loss_compartments {list} -- Which compartments to fit on (default: {['active', 'total']})

    Returns:
        dict -- dict of everything related to prediction
    """
    # record parameters for reproducability
    run_params = locals()
    run_params['model'] = model.__name__
    run_params['model_class'] = model
    
    print('Performing {} fit ..'.format('m2' if split['val_period'] == 0 else 'm1'))

    # Get data
    params = {**data}
    params['val_period'] = split['val_period']
    params['loss_compartments'] = loss['loss_compartments']
    observed_dataframes, smoothing = data_setup(**params)
    smoothing_plot = smoothing['smoothing_plot']
    orig_df_district = smoothing['df_district_unsmoothed']

    print('train\n', tabulate(observed_dataframes['df_train'].tail().round(2).T, headers='keys', tablefmt='psql'))
    if not observed_dataframes['df_val'] is None:
        print('val\n', tabulate(observed_dataframes['df_val'].tail().round(2).T, headers='keys', tablefmt='psql'))
        
    predictions_dict = run_cycle(observed_dataframes, data, model, variable_param_ranges, 
            default_params, fitting_method, fitting_method_params, split, loss)

    if smoothing_plot != None:
        predictions_dict['plots']['smoothing'] = smoothing_plot
        predictions_dict['smoothing_description'] = smoothing['smoothing_description']
    predictions_dict['df_district_unsmoothed'] = orig_df_district

    # record parameters for reproducibility
    predictions_dict['run_params'] = run_params

    return predictions_dict
