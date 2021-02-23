import os
import sys
import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from curvefit.core import functions
from pathos.multiprocessing import ProcessingPool as Pool

from data.processing.processing import get_data, train_val_test_split
from viz import plot_smoothing, plot_fit

sys.path.append('../..')

from models.ihme.model import IHME
from utils.fitting.data import lograte_to_cumulative, rate_to_cumulative
from utils.fitting.loss import Loss_Calculator
from utils.generic.enums import Columns
from main.ihme.optimiser import Optimiser
from utils.fitting.smooth_jump import smooth_big_jump


def get_rates(df, population):
    """Adds population normalized versions for each time series in the dataframe

    Args:
        df (pd.DataFrame): dataframe of time series
        population (int): population of the region

    Returns:
        pd.DataFrame: dataframe of time series with additional columns
    """
    data = df.set_index('date')
    which_columns = df.select_dtypes(include='number').columns
    for column in which_columns:
        if column.name in data.columns:
            data[f'{column.name}_rate'] = data[column.name] / population
            data[f'log_{column.name}_rate'] = data[f'{column.name}_rate'].apply(lambda x: np.log(x))
    data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])
    return data


def data_setup(data_source, dataloading_params, smooth_jump, smooth_jump_params, split,
               loss_compartments, rolling_average, rolling_average_params, population, **kwargs):
    """Helper function for single_fitting_cycle where data from different sources (given input) is imported

    Args:
        data_source ():
        dataloading_params ():
        smooth_jump ():
        smooth_jump_params ():
        split ():
        loss_compartments ():
        rolling_average ():
        rolling_average_params ():
        **kwargs ():

    Returns:

    """
    # Fetch data dictionary
    data_dict = get_data(data_source, dataloading_params)
    df_district = data_dict['data_frame']

    # Make a copy of original unsmoothed data
    orig_df_district = copy.copy(df_district)

    # Smoothing operations
    smoothing = {}
    if smooth_jump:
        # Perform smoothing
        df_district, description = smooth_big_jump(df_district, smooth_jump_params)

        # Plot smoothed data
        smoothing_plot = plot_smoothing(orig_df_district, df_district, dataloading_params['state'],
                                        dataloading_params['district'], which_compartments=loss_compartments,
                                        description='Smoothing')
        smoothing = {
            'smoothing_description': description,
            'smoothing_plot': smoothing_plot,
            'df_district_unsmoothed': orig_df_district
        }
        print(smoothing['smoothing_description'])

    # Drop rows with NA values
    df_district.dropna(axis=0, how='any', subset=['total'], inplace=True)
    df_district.reset_index(drop=True, inplace=True)

    df_district = get_rates(df_district, population)

    # Add group and covs columns
    df_district.loc[:, 'group'] = len(df_district) * [1.0]
    df_district.loc[:, 'covs'] = len(df_district) * [1.0]

    # Perform split with/without rolling average
    rap = rolling_average_params
    if rolling_average:
        df_train, df_val, df_test = train_val_test_split(
            df_district, train_period=split['train_period'], val_period=split['val_period'],
            test_period=split['test_period'], start_date=split['start_date'], end_date=split['end_date'],
            window_size=rap['window_size'], center=rap['center'],
            win_type=rap['win_type'], min_periods=rap['min_periods'])
    else:
        df_train, df_val, df_test = train_val_test_split(
            df_district, train_period=split['train_period'], val_period=split['val_period'],
            test_period=split['test_period'], start_date=split['start_date'], end_date=split['end_date'],
            window_size=1)

    df_train_nora, df_val_nora, df_test_nora = train_val_test_split(
        df_district, train_period=split['train_period'], val_period=split['val_period'],
        test_period=split['test_period'], start_date=split['start_date'], end_date=split['end_date'],
        window_size=1)

    observed_dataframes = {}
    for name in ['df_district', 'df_train', 'df_val', 'df_test', 'df_train_nora', 'df_val_nora', 'df_test_nora']:
        observed_dataframes[name] = eval(name)
    if 'ideal_params' in data_dict:
        return {"observed_dataframes": observed_dataframes, "smoothing": smoothing,
                "ideal_params": data_dict['ideal_params']}
    return {"observed_dataframes": observed_dataframes, "smoothing": smoothing}

def run_cycle():
    pass

def single_fitting_cycle():
    pass


def setup(sub_region, region, area_names, model_params, sd, smooth, test_size, smooth_jump, start_date, data_length,
          data_source, **config):
    """
    gets data and sets up the model_parameters to be ready for IHME consumption

    Args:
        sub_region (str): district
        region (str): state
        area_names (list): census area_names for the district
        model_params (dict): model_params
        sd (boolean): use social distancing covariates
        smooth (int): apply rollingavg smoothing
        test_size (int): size of test set
        smooth_jump (boolean): whether to smooth_big_jump
        start_date (str): start date for data
        data_length (int): total length of data used
        data_source (str): data source used

    Returns:
        tuple: dataframes dict, district_total_population, and model_params (modified)
    """
    model_params['func'] = getattr(functions, model_params['func'])
    model_params['covs'] = ['covs', 'sd', 'covs'] if sd else ['covs', 'covs', 'covs']
    dataframes, dtp = get_regional_data(sub_region, region, area_names, test_size=test_size, smooth_window=smooth,
                                        smooth_jump=smooth_jump, start_date=start_date, data_length=data_length,
                                        data_source=data_source, )

    return dataframes, dtp, model_params


def create_output_folder(fname):
    """
    creates folder in outputs/ihme/

    Args:
        fname (str): name of folder within outputs/ihme

    Returns:
        str: output_folder path
    """
    output_folder = f'../../outputs/ihme/{fname}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder


def forecast(model, train, test, forecast_days):
    predictions = pd.DataFrame(columns=[model.date, model.ycol])
    if len(test) == 0:
        n_days = (train[model.date].max() - train[model.date].min() + timedelta(days=1 + forecast_days)).days
        predictions.loc[:, model.ycol] = model.predict(train[model.date].min(),
                                                       train[model.date].max() + timedelta(days=forecast_days))
    else:
        n_days = (test[model.date].max() - train[model.date].min() + timedelta(days=1 + forecast_days)).days
        predictions.loc[:, model.ycol] = model.predict(train[model.date].min(),
                                                       test[model.date].max() + timedelta(days=forecast_days))
    predictions.loc[:, model.date] = pd.to_datetime(
        pd.Series([timedelta(days=x) + train[model.date].min() for x in range(n_days)]))
    return predictions


def calc_loss(ycol, train, test, predictions, xform_func, dtp):
    lc = Loss_Calculator()
    train_pred = predictions[ycol][:len(train)]
    train_model_ycol_numpy = train[ycol].to_numpy()
    test_model_ycol_numpy, test_pred = None, None
    trainerr = lc.evaluate(train_model_ycol_numpy, train_pred)
    testerr = None
    if len(test) != 0:
        test_pred = predictions[ycol][len(train):len(train) + len(test)]
        test_model_ycol_numpy = test[ycol].to_numpy()
        testerr = lc.evaluate(test_model_ycol_numpy, test_pred)
    if xform_func is not None:
        xform_trainerr = lc.evaluate(xform_func(train_model_ycol_numpy, dtp),
                                     xform_func(train_pred, dtp))
        xform_trainerr_pointwise = lc.evaluate_pointwise(xform_func(train_model_ycol_numpy, dtp),
                                                         xform_func(train_pred, dtp))
        xform_testerr = None
        xform_testerr_pointwise = None
        if len(test) != 0:
            xform_testerr = lc.evaluate(xform_func(test_model_ycol_numpy, dtp), xform_func(test_pred, dtp))
            xform_testerr_pointwise = lc.evaluate_pointwise(xform_func(test_model_ycol_numpy, dtp),
                                                            xform_func(test_pred, dtp))
    else:
        xform_trainerr, xform_testerr = None, None
        xform_trainerr_pointwise, xform_testerr_pointwise = None, None

    # CREATE POINTWISE LOSS DATAFRAME

    df_trainerr_pointwise = lc.create_pointwise_loss_dataframe(train_model_ycol_numpy, train_pred)
    df_xform_trainerr_pointwise = pd.DataFrame()
    if xform_trainerr_pointwise is not None:
        df_xform_trainerr_pointwise = lc.create_pointwise_loss_dataframe(
            xform_func(train_model_ycol_numpy, dtp), xform_func(train_pred, dtp))
    df_train_loss_pointwise = pd.concat([df_trainerr_pointwise, df_xform_trainerr_pointwise],
                                        keys=['train_no_xform', 'train']).rename_axis(['split', 'loss_functions'])
    df_train_loss_pointwise.columns = train['date'].tolist()

    df_test_loss_pointwise = pd.DataFrame()
    if len(test) != 0:
        df_testerr_pointwise = lc.create_pointwise_loss_dataframe(test_model_ycol_numpy, test_pred)
        df_xform_testerr_pointwise = pd.DataFrame()
        if xform_testerr_pointwise is not None:
            df_xform_testerr_pointwise = lc.create_pointwise_loss_dataframe(
                xform_func(test_model_ycol_numpy, dtp), xform_func(test_pred, dtp))
        df_test_loss_pointwise = pd.concat([df_testerr_pointwise, df_xform_testerr_pointwise],
                                           keys=['val_no_xform', 'val']).rename_axis(['split', 'loss_functions'])
        df_test_loss_pointwise.columns = test['date'].tolist()

    loss_dict = {
        "train": xform_trainerr,
        "val": xform_testerr,
        "train_no_xform": trainerr,
        "val_no_xform": testerr
    }

    df_loss = pd.DataFrame.from_dict(loss_dict, orient='index').stack()
    df_loss.name = 'loss'
    return df_loss, df_train_loss_pointwise, df_test_loss_pointwise


def run_cycle(dataframes, model_params, forecast_days=30,
              max_evals=1000, num_hyperopt=1, val_size=7,
              min_days=7, scoring='mape', dtp=None, xform_func=None, **kwargs):
    """
    runs a fitting cycle for 1 compartment

    Args:
        dataframes (dict): contains smoothed and unsmoothed dataframes: train, test, df
        model_params (dict): model_params
        forecast_days (int, optional): how far to predict. Defaults to 30.
        max_evals (int, optional): num evals in hyperparam optimisation. Defaults to 1000.
        num_hyperopt (int, optional): number of times to run hyperopt in parallel. Defaults to 1.
        val_size (int, optional): val size - hyperopt. Defaults to 7.
        min_days (int, optional): min train_period. Defaults to 7.
        scoring (str, optional): 'mape', 'rmse', or 'rmsle. Defaults to 'mape'.
        dtp ([type], optional): district total population. Defaults to None.
        xform_func ([type], optional): function to transform the data back to # cases. Defaults to None.

    Returns:
        dict: results_dict
    """
    model = IHME(model_params)
    train, test = dataframes['train'], dataframes['test_nora']

    n_days_optimize = kwargs.get("n_days_optimize", False)

    # OPTIMIZE HYPERPARAMS
    hyperopt_runs = {}
    trials_dict = {}
    args = {
        'bounds': copy.copy(model.priors['fe_bounds']),
        'iterations': max_evals,
        'scoring': scoring,
        'val_size': val_size,
        'min_days': min_days,
    }
    o = Optimiser(model, train, args)

    if num_hyperopt == 1:
        best_init, err, trials = o.optimisestar(0)
        hyperopt_runs[err] = best_init
        trials_dict[0] = trials
    else:
        pool = Pool(processes=5)
        for i, (best_init, err, trials) in enumerate(pool.map(o.optimisestar, list(range(num_hyperopt)))):
            hyperopt_runs[err] = best_init
            trials_dict[i] = trials
    fe_init = hyperopt_runs[min(hyperopt_runs.keys())]

    train.loc[:, 'day'] = (train['date'] - np.min(train['date'])).apply(lambda x: x.days)
    test.loc[:, 'day'] = (test['date'] - np.min(train['date'])).apply(lambda x: x.days)
    test.index = range(1 + train.index[-1], 1 + train.index[-1] + len(test))
    model.priors['fe_init'] = fe_init

    # FIT/PREDICT
    model.fit(train)

    predictions = pd.DataFrame(columns=[model.date, model.ycol])
    if len(test) == 0:
        n_days = (train[model.date].max() - train[model.date].min() + timedelta(days=1 + forecast_days)).days
        predictions.loc[:, model.ycol] = model.predict(train[model.date].min(),
                                                       train[model.date].max() + timedelta(days=forecast_days))
    else:
        n_days = (test[model.date].max() - train[model.date].min() + timedelta(days=1 + forecast_days)).days
        predictions.loc[:, model.ycol] = model.predict(train[model.date].min(),
                                                       test[model.date].max() + timedelta(days=forecast_days))
    predictions.loc[:, model.date] = pd.to_datetime(
        pd.Series([timedelta(days=x) + train[model.date].min() for x in range(n_days)]))

    # LOSS
    lc = Loss_Calculator()
    train_pred = predictions[model.ycol][:len(train)]
    train_model_ycol_numpy = train[model.ycol].to_numpy()
    test_model_ycol_numpy, test_pred = None, None
    trainerr = lc.evaluate(train_model_ycol_numpy, train_pred)
    testerr = None
    if len(test) != 0:
        test_pred = predictions[model.ycol][len(train):len(train) + len(test)]
        test_model_ycol_numpy = test[model.ycol].to_numpy()
        testerr = lc.evaluate(test_model_ycol_numpy, test_pred)
    if xform_func is not None:
        xform_trainerr = lc.evaluate(xform_func(train_model_ycol_numpy, dtp),
                                     xform_func(train_pred, dtp))
        xform_trainerr_pointwise = lc.evaluate_pointwise(xform_func(train_model_ycol_numpy, dtp),
                                                         xform_func(train_pred, dtp))
        xform_testerr = None
        xform_testerr_pointwise = None
        if len(test) != 0:
            xform_testerr = lc.evaluate(xform_func(test_model_ycol_numpy, dtp), xform_func(test_pred, dtp))
            xform_testerr_pointwise = lc.evaluate_pointwise(xform_func(test_model_ycol_numpy, dtp),
                                                            xform_func(test_pred, dtp))
    else:
        xform_trainerr, xform_testerr = None, None
        xform_trainerr_pointwise, xform_testerr_pointwise = None, None

    # CREATE POINTWISE LOSS DATAFRAME

    df_trainerr_pointwise = lc.create_pointwise_loss_dataframe(train_model_ycol_numpy, train_pred)
    df_xform_trainerr_pointwise = pd.DataFrame()
    if xform_trainerr_pointwise is not None:
        df_xform_trainerr_pointwise = lc.create_pointwise_loss_dataframe(
            xform_func(train_model_ycol_numpy, dtp), xform_func(train_pred, dtp))
    df_train_loss_pointwise = pd.concat([df_trainerr_pointwise, df_xform_trainerr_pointwise],
                                        keys=['train_no_xform', 'train']).rename_axis(['split', 'loss_functions'])
    df_train_loss_pointwise.columns = train['date'].tolist()

    df_test_loss_pointwise = pd.DataFrame()
    if len(test) != 0:
        df_testerr_pointwise = lc.create_pointwise_loss_dataframe(test_model_ycol_numpy, test_pred)
        df_xform_testerr_pointwise = pd.DataFrame()
        if xform_testerr_pointwise is not None:
            df_xform_testerr_pointwise = lc.create_pointwise_loss_dataframe(
                xform_func(test_model_ycol_numpy, dtp), xform_func(test_pred, dtp))
        df_test_loss_pointwise = pd.concat([df_testerr_pointwise, df_xform_testerr_pointwise],
                                           keys=['val_no_xform', 'val']).rename_axis(['split', 'loss_functions'])
        df_test_loss_pointwise.columns = test['date'].tolist()

    loss_dict = {
        "train": xform_trainerr,
        "val": xform_testerr,
        "train_no_xform": trainerr,
        "val_no_xform": testerr
    }

    df_loss = pd.DataFrame.from_dict(loss_dict, orient='index').stack()
    df_loss.name = 'loss'

    # UNCERTAINTY
    draws = None
    if model_params['pipeline_args']['n_draws'] > 0:
        draws_dict = model.calc_draws()
        for k in draws_dict.keys():
            low = draws_dict[k]['lower']
            up = draws_dict[k]['upper']
            # TODO: group handling
            draws = np.vstack((low, up))

    result_dict = {
        'best_params': fe_init,
        'variable_param_ranges': model.priors['fe_bounds'],
        'optimiser': o,
        'df_prediction': predictions,
        'df_district': dataframes['df'],
        'df_train': train,
        'df_val': test,
        'df_district_nora': dataframes['df_nora'],
        'df_train_nora': dataframes['train_nora'],
        'df_val_nora': dataframes['test_nora'],
        'df_loss': df_loss,
        'df_train_loss_pointwise': df_train_loss_pointwise,
        'df_test_loss_pointwise': df_test_loss_pointwise,
        'trials': trials_dict,
        'data_last_date': dataframes['df'][model.date].max(),
        'draws': draws,
        'mod.params': model.pipeline.mod.params,
        'district_total_pop': dtp,
    }

    return result_dict


def run_cycle_compartments(dataframes, model_params, which_compartments=Columns.curve_fit_compartments(),
                           forecast_days=30, max_evals=1000, num_hyperopt=1, val_size=7, min_days=7, scoring='mape',
                           dtp=None, log=True, **config):
    """
    runs fitting cycles for all compartments in which_compartments
    model_params['ycol'] is ignored here

    Args:
        dataframes (dict): contains smoothed and unsmoothed dataframes: train, test, df
        model_params (dict): model_params
        which_compartments (list, optional): List of compartments to fit. Defaults to Columns.curve_fit_compartments().
        forecast_days (int, optional): how far to predict. Defaults to 30.
        max_evals (int, optional): num evals in hyperparam optimisation. Defaults to 1000.
        num_hyperopt (int, optional): number of times to run hyperopt in parallel. Defaults to 1.
        val_size (int, optional): val size - hyperopt. Defaults to 7.
        min_days (int, optional): min train_period. Defaults to 7.
        scoring (str, optional): 'mape', 'rmse', or 'rmsle. Defaults to 'mape'.
        dtp ([type], optional): district total population. Defaults to None.
        log (bool, optional): whether to fit to log(rate). Defaults to True.

    Returns:
        dict: results_dict
    """
    xform_func = lograte_to_cumulative if log else rate_to_cumulative
    compartment_names = [col.name for col in which_compartments]
    results = {}
    ycols = {col: '{log}{colname}_rate'.format(log='log_' if log else '', colname=col.name) for col in
             which_compartments}
    loss_dict = dict()
    for i, col in enumerate(which_compartments):
        col_params = copy.copy(model_params)
        if config.get("active_log_derf", False) and col.name == 'hospitalised':
            col_params['func'] = functions.log_derf
        col_params['ycol'] = ycols[col]
        results[col.name] = run_cycle(
            dataframes, col_params, dtp=dtp, xform_func=xform_func,
            max_evals=max_evals, num_hyperopt=num_hyperopt, val_size=val_size,
            min_days=min_days, scoring=scoring, log=log, forecast_days=forecast_days,
            **config)

        # Aggregate Results
        pred = results[col.name]['df_prediction'].set_index('date')
        loss_dict[col.name] = results[col.name]['df_loss']
        if i == 0:
            predictions = pd.DataFrame(index=pred.index, columns=compartment_names + list(ycols.values()), dtype=float)

        predictions.loc[pred.index, col.name] = xform_func(pred[ycols[col]], dtp)
        predictions.loc[pred.index, ycols[col]] = pred[ycols[col]]

    df_loss = pd.concat(loss_dict.values(), axis=0, keys=compartment_names,
                        names=['compartment', 'split', 'loss_function'])
    df_loss.name = 'loss'
    df_train_loss_pointwise = pd.concat([results[comp]['df_train_loss_pointwise'] for comp in compartment_names],
                                        keys=compartment_names, names=['compartment', 'split', 'loss_function'])
    df_test_loss_pointwise = pd.concat([results[comp]['df_test_loss_pointwise'] for comp in compartment_names],
                                       keys=compartment_names, names=['compartment', 'split', 'loss_function'])

    predictions.reset_index(inplace=True)
    df_train = dataframes['train'][compartment_names + list(ycols.values()) + [model_params['date']]]
    df_val = dataframes['test'][compartment_names + list(ycols.values()) + [model_params['date']]]
    df_district = dataframes['df'][compartment_names + list(ycols.values()) + [model_params['date']]]
    df_train_nora = dataframes['train_nora'][compartment_names + list(ycols.values()) + [model_params['date']]]
    df_val_nora = dataframes['test_nora'][compartment_names + list(ycols.values()) + [model_params['date']]]
    df_district_nora = dataframes['df_nora'][compartment_names + list(ycols.values()) + [model_params['date']]]

    draws = None
    if model_params['pipeline_args']['n_draws'] > 0:
        draws = {
            col.name: {
                'draws': xform_func(results[col.name]['draws'], dtp),
                'no_xform_draws': results[col.name]['draws'],
            } for col in which_compartments
        }

    final = {
        'best_params': {col.name: results[col.name]['best_params'] for col in which_compartments},
        'variable_param_ranges': model_params['priors']['fe_bounds'],
        'n_days': {col.name: results[col.name]['n_days'] for col in which_compartments},
        'df_prediction': predictions,
        'df_district': df_district,
        'df_train': df_train,
        'df_val': df_val,
        'df_district_nora': df_district_nora,
        'df_train_nora': df_train_nora,
        'df_val_nora': df_val_nora,
        'df_loss': df_loss,
        'df_train_loss_pointwise': df_train_loss_pointwise,
        'df_test_loss_pointwise': df_test_loss_pointwise,
        'data_last_date': df_district[model_params['date']].max(),
        'draws': draws,
        'mod.params': {col.name: results[col.name]['mod.params'] for col in which_compartments},
        'individual_results': results,
        'district_total_pop': results[which_compartments[0].name]['district_total_pop'],
    }

    return final


def single_cycle(sub_region, region, area_names=None, model_params=None,
                 which_compartments=Columns.curve_fit_compartments(), **config):
    """[summary]

    Args:
        sub_region (str): district
        region (str): state
        area_names (list): census area_names for the district
        model_params (dict): model_params
        which_compartments (list, optional): List of compartments to fit. Defaults to Columns.curve_fit_compartments().

    Returns:
        dict: results_dict
    """
    dataframes, dtp, model_params = setup(sub_region, region, area_names, model_params, **config)
    return run_cycle_compartments(dataframes, model_params, which_compartments=which_compartments, dtp=dtp, **config)
