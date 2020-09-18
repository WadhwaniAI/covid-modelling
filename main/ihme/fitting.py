import os
import sys
from copy import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from curvefit.core import functions
from pathos.multiprocessing import ProcessingPool as Pool

from data.processing.processing import train_val_test_split

sys.path.append('../..')
from utils.data import get_rates
from data.processing import get_data_from_source
from utils.util import get_subset

from models.ihme.model import IHME
from utils.data import lograte_to_cumulative, rate_to_cumulative
from utils.loss import Loss_Calculator
from utils.enums import Columns
from main.ihme.optimiser import Optimiser
from utils.smooth_jump import smooth_big_jump


def preprocess(timeseries, region, sub_region=None, area_names=None, trim_deceased=False):
    df, dtp = get_rates(timeseries, region, sub_region=sub_region, area_names=area_names)
    df.loc[:, 'sd'] = df['date'].apply(lambda x: [1.0 if x >= datetime(2020, 3, 24) else 0.0]).tolist()
    if trim_deceased:
        df = df.loc[df['deceased_rate'].gt(1e-15).idxmax():, :].reset_index()
    df.loc[:, 'day'] = (df['date'] - np.min(df['date'])).apply(lambda x: x.days)
    return df, dtp


def get_regional_data(sub_region, region, area_names, test_size, smooth_window, smooth_jump, start_date=None,
                      data_length=0, data_source='covid19india'):
    """
    Function to get regional data and shape it for IHME consumption

    Args:
        sub_region (str): district
        region (str): state
        area_names (list): census area_names for the district
        test_size (int): size of test set
        smooth_window (int): apply rollingavg smoothing
        smooth_jump (bool): whether to smooth_big_jump
        start_date (str, optional): start date for data
        data_length (int, optional): total length of data used
        data_source (str, optional): data source used (default: covid19india)

    Returns:
        dict: contains smoothed and unsmoothed dataframes: train, test, df
    """
    timeseries, df = dict(), dict()
    timeseries['df'] = get_data_from_source(region=region, sub_region=sub_region, data_source=data_source)
    timeseries['df_nora'] = smooth_big_jump(timeseries['df']) if smooth_jump else copy(timeseries['df'])
    col_names = [c.name for c in Columns.curve_fit_compartments()]
    col_names.extend([f'{c.name}_rate' for c in Columns.curve_fit_compartments()])
    col_names.extend([f'log_{c.name}_rate' for c in Columns.curve_fit_compartments()])

    for df_type in ['df', 'df_nora']:
        df[df_type], dtp = preprocess(timeseries[df_type], region, sub_region=sub_region, area_names=area_names)
        start_date = pd.to_datetime(start_date, dayfirst=False) if start_date is not None else df['date'].min()
        df[df_type] = get_subset(
            df[df_type], lower=start_date, upper=start_date+timedelta(data_length-1), col='date').reset_index(drop=True)

    df['train'], _, df['test'] = train_val_test_split(df['df'], val_size=0, test_size=test_size,
                                                      rolling_window=smooth_window, end='actual', dropna=False,
                                                      train_rollingmean=True, val_rollingmean=True,
                                                      test_rollingmean=True, which_columns=col_names)
    df['train_nora'], _, df['test_nora'] = train_val_test_split(df['df_nora'], val_size=0, test_size=test_size,
                                                                rolling_window=smooth_window, end='actual', dropna=False,
                                                                train_rollingmean=False, val_rollingmean=False,
                                                                test_rollingmean=False, which_columns=col_names)

    return df, dtp


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
        'bounds': copy(model.priors['fe_bounds']),
        'iterations': max_evals,
        'scoring': scoring,
        'val_size': val_size,
        'min_days': min_days,
    }
    o = Optimiser(model, train, args)

    if num_hyperopt == 1:
        (best_init, n_days), err, trials = o.optimisestar(0)
        hyperopt_runs[err] = (best_init, n_days)
        trials_dict[0] = trials
    else:
        pool = Pool(processes=5)
        for i, ((best_init, n_days), err, trials) in enumerate(pool.map(o.optimisestar, list(range(num_hyperopt)))):
            hyperopt_runs[err] = (best_init, n_days)
            trials_dict[i] = trials
    (fe_init, n_days_train) = hyperopt_runs[min(hyperopt_runs.keys())]

    if n_days_optimize:
        train = train[-n_days_train:]
    train.loc[:, 'day'] = (train['date'] - np.min(train['date'])).apply(lambda x: x.days)
    test.loc[:, 'day'] = (test['date'] - np.min(train['date'])).apply(lambda x: x.days)
    if n_days_optimize:
        train.reset_index(inplace=True)
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
        'n_days': n_days_train,
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
                           forecast_days=30,
                           max_evals=1000, num_hyperopt=1, val_size=7,
                           min_days=7, scoring='mape', dtp=None, xform_func=None, log=True, **config):
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
        xform_func ([type], optional): function to transform the data back to # cases. Defaults to None.
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
        col_params = copy(model_params)
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
    return run_cycle_compartments(dataframes, model_params, dtp=dtp, which_compartments=which_compartments, **config)
