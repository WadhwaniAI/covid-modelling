import os
import sys
import pandas as pd
import numpy as np
from copy import copy
from datetime import datetime, timedelta
from pathos.multiprocessing import ProcessingPool as Pool

from curvefit.core import functions

sys.path.append('../..')
from utils.data import get_rates
from data.processing import get_district_timeseries_cached
from utils.util import train_test_split, rollingavg

from models.ihme.model import IHME
from utils.data import lograte_to_cumulative, rate_to_cumulative
from utils.loss import Loss_Calculator
from utils.enums import Columns
from main.ihme.optimiser import Optimiser
from utils.smooth_jump import smooth_big_jump


def get_regional_data(dist, st, area_names, ycol, test_size, smooth_window, disable_tracker,
                      smooth_jump, smooth_jump_method, smooth_jump_days, start_date=None, dataset_length=0,
                      continuous_ra=True):
    """
    Function to get regional data and shape it for IHME consumption

    Args:
        dist ([type]): district
        st ([type]): state
        area_names ([type]): census area_names for the district
        ycol ([type]): name of ycol
        test_size ([type]): size of test set
        smooth_window ([type]): apply rollingavg smoothing
        disable_tracker ([type]): disable covid19api, use athena instead
        smooth_jump (boolean): whether to smooth_big_jump
        smooth_jump_method ([type]): passed to smooth_big_jump
        smooth_jump_days ([type]): passed to smooth_big_jump
        start_date ():
        dataset_length ():
        continuous_ra ():

    Returns:
        dict: contains smoothed and unsmoothed dataframes: train, test, df
    """    
    district_timeseries_nora = get_district_timeseries_cached(
        dist, st, disable_tracker=disable_tracker)
    if smooth_jump:
        district_timeseries = smooth_big_jump(district_timeseries_nora, not disable_tracker, method=smooth_jump_method)
    else:
        district_timeseries = copy(district_timeseries_nora)

    df_nora, _ = get_rates(district_timeseries_nora, st, area_names)
    df, dtp = get_rates(district_timeseries, st, area_names)
    
    df.loc[:,'sd'] = df['date'].apply(lambda x: [1.0 if x >= datetime(2020, 3, 24) else 0.0]).tolist()
    df_nora.loc[:,'sd'] = df_nora['date'].apply(lambda x: [1.0 if x >= datetime(2020, 3, 24) else 0.0]).tolist()

    col_names = [c.name for c in Columns.curve_fit_compartments()]
    if smooth_window > 0:
        for col in col_names:
            df[col] = rollingavg(df[col], smooth_window)
    offset = smooth_window//2
    
    startday = df['date'][df['deceased_rate'].gt(1e-15).idxmax()]
    
    df = df.loc[df['deceased_rate'].gt(1e-15).idxmax():, :].reset_index()
    df_nora = df_nora.loc[df_nora['deceased_rate'].gt(1e-15).idxmax():,:].reset_index()

    df.loc[:, 'day'] = (df['date'] - np.min(df['date'])).apply(lambda x: x.days)
    df_nora.loc[:, 'day'] = (df_nora['date'] - np.min(df_nora['date'])).apply(lambda x: x.days)

    if start_date is None:
        start_date = df['date'].min()
    else:
        start_date = pd.to_datetime(start_date, dayfirst=False)

    threshold = start_date - timedelta(1)

    _, df_ = train_test_split(df, threshold)
    _, df_nora_ = train_test_split(df_nora, threshold)

    if dataset_length == 0:
        dataset_length = df.shape[0]

    df_ = df_.head(dataset_length)
    df_nora_ = df_nora_.head(dataset_length)

    threshold = df_['date'].max() - timedelta(days=test_size)
    
    train, test = train_test_split(df_, threshold)
    train_nora, test_nora = train_test_split(df_nora_, threshold)
    if not continuous_ra:
        train = pd.concat([train.iloc[:-2, :], train_nora.iloc[-2:, :]])

    dataframes = {
        'train': train,
        'test': test,
        'df': df,
        'train_nora': train_nora,
        'test_nora': test_nora,
        'df_nora': df_nora,
    }
    return dataframes, dtp
    
def setup(dist, st, area_names, model_params,
          sd, smooth, test_size, disable_tracker,
          smooth_jump, smooth_jump_method, smooth_jump_days, start_date, dataset_length, continuous_ra, **config):
    """
    gets data and sets up the model_parameters to be ready for IHME consumption

    Args:
        dist (str): district
        st (str): state
        area_names (list): census area_names for the district
        model_params (dict): model_params
        sd (boolean): use social distancing covariates
        smooth (int): apply rollingavg smoothing
        test_size (int): size of test set
        disable_tracker (boolean): disable covid19api, use athena instead
        smooth_jump (boolean): whether to smooth_big_jump
        smooth_jump_method ([type]): passed to smooth_big_jump
        smooth_jump_days ([type]): passed to smooth_big_jump
        start_date ():
        dataset_length ():
        continuous_ra ():

    Returns:
        tuple: dataframes dict, district_total_population, and model_params (modified)
    """    
    model_params['func'] = getattr(functions, model_params['func'])
    model_params['covs'] = ['covs', 'sd', 'covs'] if sd else ['covs', 'covs', 'covs']
    dataframes, dtp = get_regional_data(
        dist, st, area_names, ycol=model_params['ycol'], 
        smooth_window=smooth, test_size=test_size,
        disable_tracker=disable_tracker,
        smooth_jump=smooth_jump, smooth_jump_method=smooth_jump_method, 
        smooth_jump_days=smooth_jump_days,
        start_date=start_date, dataset_length=dataset_length,
        continuous_ra=continuous_ra
    )
        
    return dataframes, dtp, model_params

def create_output_folder(fname):
    """
    creates folder in outputs/ihme/

    Args:
        fname (ste): name of folder within outputs/ihme

    Returns:
        str: output_folder path
    """    
    output_folder = f'../../outputs/ihme/{fname}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

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
    train, test = dataframes['train'], dataframes['test']

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
        n_days = (train[model.date].max() - train[model.date].min() + timedelta(days=1+forecast_days)).days
        predictions.loc[:, model.ycol] = model.predict(train[model.date].min(),
            train[model.date].max() + timedelta(days=forecast_days))
    else:
        n_days = (test[model.date].max() - train[model.date].min() + timedelta(days=1+forecast_days)).days
        predictions.loc[:, model.ycol] = model.predict(train[model.date].min(),
            test[model.date].max() + timedelta(days=forecast_days))
    predictions.loc[:, model.date] = pd.to_datetime(pd.Series([timedelta(days=x)+train[model.date].min() for x in range(n_days)]))

    # LOSS
    lc = Loss_Calculator()
    train_pred = predictions[model.ycol][:len(train)]
    train_model_ycol_numpy = train[model.ycol].to_numpy()
    trainerr = lc.evaluate(train_model_ycol_numpy, train_pred)
    testerr = None
    if len(test) != 0:
        test_pred = predictions[model.ycol][len(train):len(train) + len(test)]
        test_model_ycol_numpy = test[model.ycol].to_numpy()
        testerr = lc.evaluate(test_model_ycol_numpy, test_pred)
    if xform_func != None:
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

    df_trainerr_pointwise = lc.create_pointwise_loss_dataframes(train_model_ycol_numpy, train_pred)
    df_xform_trainerr_pointwise = pd.DataFrame()
    if xform_trainerr_pointwise is not None:
        df_xform_trainerr_pointwise = lc.create_pointwise_loss_dataframes(
            xform_func(train_model_ycol_numpy, dtp), xform_func(train_pred, dtp))
    df_train_loss_pointwise = pd.concat([df_trainerr_pointwise, df_xform_trainerr_pointwise],
        keys=['train', 'train_no_xform']).rename_axis(['split', 'loss_functions'])
    df_train_loss_pointwise.columns = train['date'].tolist()

    df_test_loss_pointwise = pd.DataFrame()
    if len(test) != 0:
        df_testerr_pointwise = lc.create_pointwise_loss_dataframes(test_model_ycol_numpy, test_pred)
        df_xform_testerr_pointwise = pd.DataFrame()
        if xform_testerr_pointwise is not None:
            df_xform_testerr_pointwise = lc.create_pointwise_loss_dataframes(
                xform_func(test_model_ycol_numpy, dtp), xform_func(test_pred, dtp))
        df_test_loss_pointwise = pd.concat([df_testerr_pointwise, df_xform_testerr_pointwise],
            keys=['val', 'val_no_xform']).rename_axis(['split', 'loss_functions'])
        df_test_loss_pointwise.columns = test['date'].tolist()

    # UNCERTAINTY
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
        'df_loss': pd.DataFrame({
            "train": xform_trainerr,
            "val": xform_testerr,
            "train_no_xform": trainerr,
            "val_no_xform": testerr,
        }),
        'df_train_loss_pointwise': df_train_loss_pointwise,
        'df_test_loss_pointwise': df_test_loss_pointwise,
        'trials': trials_dict,
        'data_last_date': dataframes['df'][model.date].max(),
        'draws': draws,
        'mod.params': model.pipeline.mod.params,
        'district_total_pop': dtp,
    }

    # for name in ['data_from_tracker', 
    #              'df_prediction', 'df_district', 'df_train', 'df_val', 'df_loss', 'ax', 'trials', 'data_last_date']:
    #     result_dict[name] = eval(name)


    return result_dict

def run_cycle_compartments(dataframes, model_params, which_compartments=Columns.curve_fit_compartments(), forecast_days=30, 
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
    ycols = {col: '{log}{colname}_rate'.format(log='log_' if log else '', colname=col.name) for col in which_compartments}
    for i, col in enumerate(which_compartments):
        col_params = copy(model_params)
        col_params['ycol'] = ycols[col]
        results[col.name] = run_cycle(
            dataframes, col_params, dtp=dtp, xform_func=xform_func, 
            max_evals=max_evals, num_hyperopt=num_hyperopt, val_size=val_size,
            min_days=min_days, scoring=scoring, log=log, forecast_days=forecast_days,
            **config)

        # Aggregate Results
        pred = results[col.name]['df_prediction'].set_index('date')
        loss_functions = results[col.name]['df_loss'].index.tolist()
        if i == 0:
            loss_index = pd.MultiIndex.from_product([compartment_names, loss_functions],
                                                    names=['compartment', 'loss_function'])
            predictions = pd.DataFrame(index=pred.index, columns=compartment_names + list(ycols.values()), dtype=float)
            df_loss = pd.DataFrame(index=loss_index, columns=results[col.name]['df_loss'].columns)
        for loss_function in loss_functions:
            df_loss.loc[(col.name, loss_function), :] = results[col.name]['df_loss'].loc[loss_function, :]

        predictions.loc[pred.index, col.name] = xform_func(pred[ycols[col]], dtp)
        predictions.loc[pred.index, ycols[col]] = pred[ycols[col]]

    df_train_loss_pointwise = pd.concat([results[comp]['df_train_loss_pointwise'] for comp in compartment_names],
        keys=compartment_names)
    df_test_loss_pointwise = pd.concat([results[comp]['df_test_loss_pointwise'] for comp in compartment_names],
        keys=compartment_names)

    predictions.reset_index(inplace=True)
    df_train = dataframes['train'][compartment_names + list(ycols.values()) + [model_params['date']]]
    df_val = dataframes['test'][compartment_names + list(ycols.values()) + [model_params['date']]]
    df_district = dataframes['df'][compartment_names + list(ycols.values()) + [model_params['date']]]
    df_train_nora = dataframes['train_nora'][compartment_names + list(ycols.values()) + [model_params['date']]]
    df_val_nora = dataframes['test_nora'][compartment_names + list(ycols.values()) + [model_params['date']]]
    df_district_nora = dataframes['df_nora'][compartment_names + list(ycols.values()) + [model_params['date']]]
    
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
        'draws': {
            col.name: {
                'draws': xform_func(results[col.name]['draws'], dtp),
                'no_xform_draws': results[col.name]['draws'],
            } for col in which_compartments
        },
        'mod.params': {col.name: results[col.name]['mod.params'] for col in which_compartments},
        'individual_results': results,
        'district_total_pop': results[which_compartments[0].name]['district_total_pop'],
    }

    return final  

def single_cycle(dist, st, area_names, model_params, which_compartments=Columns.curve_fit_compartments(), **config):
    """[summary]

    Args:
        dist (str): district
        st (str): state
        area_names (list): census area_names for the district
        model_params (dict): model_params
        which_compartments (list, optional): List of compartments to fit. Defaults to Columns.curve_fit_compartments().

    Returns:
        dict: results_dict
    """
    dataframes, dtp, model_params = setup(dist, st, area_names, model_params, **config)
    return run_cycle_compartments(dataframes, model_params, dtp=dtp, which_compartments=which_compartments, **config)