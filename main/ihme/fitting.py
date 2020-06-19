import os
import sys
import json
import pandas as pd
import numpy as np
from copy import copy
from datetime import datetime, timedelta
from pathos.multiprocessing import ProcessingPool as Pool

from curvefit.core import functions

sys.path.append('../..')
from models.ihme.model import IHME
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
            smooth_jump, smooth_jump_method, smooth_jump_days):
    district_timeseries_nora = get_district_timeseries_cached(
        dist, st, disable_tracker=disable_tracker)
    if smooth_jump:
        district_timeseries = smooth_big_jump(district_timeseries_nora, smooth_jump_days, not disable_tracker, method=smooth_jump_method)
    df_nora, _ = get_rates(district_timeseries_nora, st, area_names)
    df, dtp = get_rates(district_timeseries, st, area_names)
    
    df.loc[:,'sd'] = df['date'].apply(lambda x: [1.0 if x >= datetime(2020, 3, 24) else 0.0]).tolist()
    df_nora.loc[:,'sd'] = df_nora['date'].apply(lambda x: [1.0 if x >= datetime(2020, 3, 24) else 0.0]).tolist()

    if smooth_window > 0:
        df[ycol] = rollingavg(df[ycol], smooth_window)
        df = df.dropna(subset=[ycol])
    
    startday = df['date'][df['deceased_rate'].gt(1e-15).idxmax()]
    
    df = df.loc[df['deceased_rate'].gt(1e-15).idxmax():,:].reset_index()
    df_nora = df_nora.loc[df_nora['deceased_rate'].gt(1e-15).idxmax():,:].reset_index()

    df.loc[:, 'day'] = (df['date'] - np.min(df['date'])).apply(lambda x: x.days)
    df_nora.loc[:, 'day'] = (df_nora['date'] - np.min(df_nora['date'])).apply(lambda x: x.days)
    
    threshold = df['date'].max() - timedelta(days=test_size)
    
    train, test = train_test_split(df, threshold)
    train_nora, test_nora = train_test_split(df_nora, threshold)
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
        smooth_jump, smooth_jump_method, smooth_jump_days, **config):
    model_params['func'] = getattr(functions, model_params['func'])
    model_params['covs'] = ['covs', 'sd', 'covs'] if sd else ['covs', 'covs', 'covs']
    dataframes, dtp = get_regional_data(
        dist, st, area_names, ycol=model_params['ycol'], 
        smooth_window=smooth, test_size=test_size,
        disable_tracker=disable_tracker,
        smooth_jump=smooth_jump, smooth_jump_method=smooth_jump_method, 
        smooth_jump_days=smooth_jump_days)
        
    return dataframes, dtp, model_params

def create_output_folder(fname):
    output_folder = f'../../outputs/ihme/{fname}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

def run_cycle(dataframes, model_params, forecast_days=30, 
    max_evals=1000, num_hyperopt=1, val_size=7,
    min_days=7, scoring='mape', dtp=None, xform_func=None, **kwargs):
    model = IHME(model_params)
    train, test = dataframes['train'], dataframes['test']

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
    
    train = train[-n_days_train:]
    train.loc[:, 'day'] = (train['date'] - np.min(train['date'])).apply(lambda x: x.days)
    test.loc[:, 'day'] = (test['date'] - np.min(train['date'])).apply(lambda x: x.days)
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
    trainerr = lc.evaluate(train[model.ycol], train_pred)
    testerr = None
    if len(test) != 0:
        test_pred = predictions[model.ycol][len(train):len(train) + len(test)]
        testerr = lc.evaluate(test[model.ycol], test_pred)
    if xform_func != None:
        xform_trainerr = lc.evaluate(xform_func(train[model.ycol], dtp),
            xform_func(train_pred, dtp))
        xform_testerr = None
        if len(test) != 0:
            xform_testerr = lc.evaluate(xform_func(test[model.ycol], dtp),
                xform_func(test_pred, dtp))
    else:
        xform_trainerr, xform_testerr = None, None
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
            'train': xform_trainerr,
            "val": xform_testerr,
            "train_no_xform": trainerr,
            "val_no_xform": testerr,
        }),
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
        if i == 0:
            predictions = pd.DataFrame(index=pred.index, columns=compartment_names + list(ycols.values()), dtype=float)
            df_loss = pd.DataFrame(index=compartment_names, columns=results[col.name]['df_loss'].columns)
        df_loss.loc[col.name, :] = results[col.name]['df_loss'].loc['mape', :]
        predictions.loc[pred.index, col.name] = xform_func(pred[ycols[col]], dtp)
        predictions.loc[pred.index, ycols[col]] = pred[ycols[col]]

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
    dataframes, dtp, model_params = setup(dist, st, area_names, model_params, **config)
    return run_cycle_compartments(dataframes, model_params, dtp=dtp, which_compartments=which_compartments, **config)