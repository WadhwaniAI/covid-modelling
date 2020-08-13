import os
import json
import numpy as np
import pandas as pd

from collections import OrderedDict, defaultdict
import datetime
import copy

from data.processing.processing import get_data, train_val_split
from data.processing import granular

from models.seir import SEIRHD
from main.seir.optimiser import Optimiser
from utils.loss import Loss_Calculator
from utils.enums import Columns
from utils.smooth_jump import smooth_big_jump, smooth_big_jump_stratified
from viz import plot_smoothing, plot_fit


def get_regional_data(state, district, data_from_tracker, data_format, filename, loss_compartments,
                      granular_data=False, smooth_jump=False, return_extra=False):
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
                                        loss_compartments=loss_compartments, description=f'Smoothing')
    
    df_district['daily_cases'] = df_district['total'].diff()
    df_district.dropna(axis=0, how='any', inplace=True)
    df_district.reset_index(drop=True, inplace=True)

    if return_extra:
        extra = {
            'smoothing_description': description,
            'smoothing_plot': smoothing_plot,
            'df_district_unsmoothed': orig_df_district
        }
        print(extra['smoothing_description'])
        return df_district, extra 
    return df_district 

def data_setup(df_district, val_period):
    """Helper function for single_fitting_cycle which sets up the data including doing the train val split

    Arguments:
        df_district {pd.DataFrame} -- True observations from districts_daily/custom file/athena DB
        val_period {int} -- Length of val period

    Returns:
        dict(pd.DataFrame) -- Dict of pd.DataFrame objects
    """
    # Get train val split 
    df_train, df_val = train_val_split(
        df_district, train_rollingmean=True, val_rollingmean=True, val_size=val_period, rolling_window=7)
    df_train_nora, df_val_nora = train_val_split(
        df_district, train_rollingmean=False, val_rollingmean=False, val_size=val_period)

    observed_dataframes = {}
    for name in ['df_district', 'df_train', 'df_val', 'df_train_nora', 'df_val_nora']:
        observed_dataframes[name] = eval(name)
    return observed_dataframes


def run_cycle(state, district, observed_dataframes, model=SEIRHD, variable_param_ranges=None, 
              default_params=None, train_period=7, data_from_tracker=True,
              loss_compartments=['active', 'total', 'recovered', 'deceased'], 
              num_evals=1500, N=1e7, test_period=0):
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
    if district is None:
        district = ''

    df_district, df_train, df_val, df_train_nora, df_val_nora = [
        observed_dataframes.get(k) for k in observed_dataframes.keys()]

    # Initialise Optimiser
    optimiser = Optimiser()
    # Get the fixed params
    std_default_params = optimiser.init_default_params(df_train, N=N, train_period=train_period)
    if default_params is not None:
        default_params = {**std_default_params, **default_params}
    else:
        default_params = std_default_params
    # Get/create searchspace of variable paramms
    if test_period == 0:
        loss_indices = [-train_period, None]
    else:
        loss_indices = [-(train_period+test_period), -test_period]
    # Perform Bayesian Optimisation
    total_days = (df_train.iloc[-1, :]['date'] - default_params['starting_date']).days
    best_params, trials = optimiser.bayes_opt(df_train, default_params, variable_param_ranges, model=model, 
                                              num_evals=num_evals, loss_indices=loss_indices, method='mape',
                                              total_days=total_days, loss_compartments=loss_compartments)
    print('best parameters\n', best_params)

    if not isinstance(df_val, pd.DataFrame):
        df_prediction = optimiser.solve(best_params, default_params, df_train, end_date=df_train.iloc[-1, :]['date'],
                                        model=model)
    else:
        df_prediction = optimiser.solve(best_params, default_params, df_train, end_date=df_val.iloc[-1, :]['date'], 
                                        model=model)
    
    lc = Loss_Calculator()
    df_loss = lc.create_loss_dataframe_region(df_train_nora, df_val_nora, df_prediction, train_period, 
                                              loss_compartments=loss_compartments)

    fit_plot = plot_fit(df_prediction, df_train, df_val, df_district, train_period, state, district,
                        loss_compartments=loss_compartments)

    results_dict = {}
    results_dict['plots'] = {}
    results_dict['plots']['fit'] = fit_plot
    data_last_date = df_district.iloc[-1]['date'].strftime("%Y-%m-%d")
    for name in ['data_from_tracker', 'best_params', 'default_params', 'variable_param_ranges', 'optimiser', 
                 'df_prediction', 'df_district', 'df_train', 'df_val', 'df_loss', 'trials', 'data_last_date']:
        results_dict[name] = eval(name)

    return results_dict


def single_fitting_cycle(state, district, model=SEIRHD, variable_param_ranges=None, default_params=None, #Main 
                         data_from_tracker=True, granular_data=False, filename=None, data_format='new', #Data
                         train_period=7, val_period=7, num_evals=1500, N=1e7, test_period=0,  #Misc
                         loss_compartments=['active', 'total'], #Compartments
                         smooth_jump=False): #Smoothing
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
    
    print('Performing {} fit ..'.format('m2' if val_period == 0 else 'm1'))

    # Get data
    df_district, extra = get_regional_data(
        state, district, data_from_tracker, data_format, filename, 
        loss_compartments=loss_compartments, granular_data=granular_data,
        smooth_jump=smooth_jump, return_extra=True
    )
    smoothing_plot = extra['smoothing_plot']
    orig_df_district = extra['df_district_unsmoothed']

    # Process the data to get rolling averages and other stuff
    observed_dataframes = data_setup(df_district, val_period)

    print('train\n', observed_dataframes['df_train'].tail())
    print('val\n', observed_dataframes['df_val'])
    
    predictions_dict = run_cycle(
        state, district, observed_dataframes, 
        model=model, variable_param_ranges=variable_param_ranges, default_params=default_params,
        data_from_tracker=data_from_tracker, train_period=train_period, 
        loss_compartments=loss_compartments, N=N, test_period=test_period,
        num_evals=num_evals
    )

    if smoothing_plot != None:
        predictions_dict['plots']['smoothing'] = smoothing_plot
        predictions_dict['smoothing_description'] = extra['smoothing_description']
    predictions_dict['df_district_unsmoothed'] = orig_df_district

    # record parameters for reproducibility
    predictions_dict['run_params'] = run_params

    return predictions_dict
