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
from models.ihme.util import get_rates
from data.processing import get_district_timeseries_cached
from utils.util import train_test_split, rollingavg


from models.ihme.model import IHME
from models.ihme.util import lograte_to_cumulative, rate_to_cumulative
from utils.loss import Loss_Calculator
from utils.enums import Columns
from main.ihme.optimiser import Optimiser
from main.seir.fitting import smooth_big_jump

def get_regional_data(dist, st, area_names, ycol, test_size, smooth_window, disable_tracker):
    district_timeseries = get_district_timeseries_cached(
        dist, st, disable_tracker=disable_tracker)
    district_timeseries = smooth_big_jump(district_timeseries, 33, not disable_tracker, method='weighted')
    df, dtp = get_rates(district_timeseries, st, area_names)
    
    df.loc[:,'sd'] = df['date'].apply(lambda x: [1.0 if x >= datetime(2020, 3, 24) else 0.0]).tolist()

    smoothedcol = f'{ycol}_smoothed'
    if smooth_window > 0:
        df[smoothedcol] = rollingavg(df[ycol], smooth_window)
        df = df.dropna(subset=[smoothedcol])
    
    startday = df['date'][df['deceased_rate'].gt(1e-15).idxmax()]
    df = df.loc[df['deceased_rate'].gt(1e-15).idxmax():,:]
    df = df.reset_index()
    df.loc[:, 'day'] = (df['date'] - np.min(df['date'])).apply(lambda x: x.days)
    threshold = df['date'].max() - timedelta(days=test_size)
    train, test = train_test_split(df, threshold)
    dataframes = {
        'train': train,
        'test': test,
        'df': df,
    }
    return dataframes, dtp, smoothedcol
    
def setup(dist, st, area_names, model_params, 
        sd, smooth, test_size, disable_tracker, **config):
    model_params['func'] = getattr(functions, model_params['func'])
    model_params['covs'] = ['covs', 'sd', 'covs'] if sd else ['covs', 'covs', 'covs']
    dataframes, dtp, smoothedcol = get_regional_data(
        dist, st, area_names, ycol=model_params['ycol'], 
        smooth_window=smooth, test_size=test_size,
        disable_tracker=disable_tracker)
        
    if smooth > 0:
        model_params['ycol'] = smoothedcol
    
    return dataframes, dtp, model_params

def create_output_folder(fname):
    today = datetime.today()
    output_folder = f'../../outputs/ihme/{fname}/{today}'
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
        'fe_init': fe_init,
        'n_days': n_days_train,
        'error': {
            'train': trainerr,
            "test": testerr
        },
        'xform_error': {
            'train': xform_trainerr,
            "test": xform_testerr
        },
        'predictions': {
            'start': train[model.date].min(),
            'fit_dates': train[model.date],
            'val_dates': test[model.date],
            'future_dates': predictions[model.date][len(train) + len(test):],
            'predictions': predictions.set_index(model.date),
        },
        'trials': trials_dict,
        'draws': draws,
        'mod.params': model.pipeline.mod.params,
        'train': train,
        'test': test,
        'df': dataframes['df'],
        'district_total_pop': dtp,
    }

    return result_dict

def single_cycle(dist, st, area_names, model_params, **config):
    dataframes, dtp, model_params = setup(dist, st, area_names, model_params, **config)
    # create_output_folder(dist)
    xform_func = lograte_to_cumulative if config['log'] else rate_to_cumulative
    return run_cycle(dataframes, model_params, dtp=dtp, xform_func=xform_func, **config)

def single_cycle_multiple(dist, st, area_names, model_params, which_compartments=Columns.which_compartments(), **config):
    dataframes, dtp, model_params = setup(dist, st, area_names, model_params, **config)
    xform_func = lograte_to_cumulative if config['log'] else rate_to_cumulative
    results = {}
    
    for col in which_compartments:
        col_params = copy(model_params)
        col_params['ycol'] = '{log}{colname}_rate'.format(log='log_' if config['log'] else '', colname=col.name)
        results[col.name] = run_cycle(dataframes, col_params, dtp=dtp, xform_func=xform_func, **config)

    base = results[which_compartments[0].name]
    
    predictions = pd.DataFrame(index=results['deceased']['predictions']['predictions'].index)
    for key in results.keys():
        pred = results[key]['predictions']['predictions']
        predictions.loc[pred.index, key] = pred['{log}{colname}_rate'.format(log='log_' if config['log'] else '', colname=key)]
    
    predictions = xform_func(predictions, dtp)

    final = {
        'individual_results': results,
        'predictions': {
            'start': base['predictions']['start'],
            'fit_dates': base['predictions']['fit_dates'],
            'val_dates': base['predictions']['val_dates'],
            'future_dates': base['predictions']['future_dates'],
            'predictions': predictions,
        },
        'train': base['train'],
        'test': base['test'],
        'df': base['df'],
        'district_total_pop': base['district_total_pop'],
    }

    return final    