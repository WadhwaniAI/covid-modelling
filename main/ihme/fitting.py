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
from models.ihme.util import get_mortality
from data.processing import get_district_timeseries_cached
from utils.util import train_test_split, rollingavg


from models.ihme.model import IHME
from models.ihme.util import lograte_to_cumulative, rate_to_cumulative
from utils.loss import Loss_Calculator

from main.ihme.optimiser import Optimiser

def get_regional_data(dist, st, area_names, ycol, test_size, smooth_window, disable_tracker):
    district_timeseries = get_district_timeseries_cached(dist, st, disable_tracker=disable_tracker)
    df, dtp = get_mortality(district_timeseries, st, area_names)
    
    df.loc[:,'sd'] = df['date'].apply(lambda x: [1.0 if x >= datetime(2020, 3, 24) else 0.0]).tolist()

    smoothedcol = f'{ycol}_smoothed'
    if smooth_window > 0:
        df[smoothedcol] = rollingavg(df[ycol], smooth_window)
        df = df.dropna(subset=[smoothedcol])
    
    startday = df['date'][df['mortality'].gt(1e-15).idxmax()]
    df = df.loc[df['mortality'].gt(1e-15).idxmax():,:]
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
    
def setup(dist, st, area_names, config, model_params):
    model_params['func'] = getattr(functions, model_params['func'])
    model_params['covs'] = ['covs', 'sd', 'covs'] if config['sd'] else ['covs', 'covs', 'covs']
    dataframes, dtp, smoothedcol = get_regional_data(
        dist, st, area_names, ycol=model_params['ycol'], 
        smooth_window=config['smooth'], test_size=config['test_size'],
        disable_tracker=config['disable_tracker'])
        
    if config['smooth'] > 0:
        model_params['ycol'] = smoothedcol
    
    fname = f'{dist}_deaths'
    return dataframes, dtp, model_params, fname

def create_output_folder(fname):
    today = datetime.today()
    output_folder = f'../../outputs/ihme/{fname}/{today}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

def run_cycle(dataframes, model_params, predict_days=30, 
    max_evals=1000, num_hyperopt_runs=1, val_size=7,
    min_days=7, scoring='mape', dtp=None, xform_func=None):
    model = IHME(model_params)
    train, test = dataframes['train'], dataframes['test']

    # OPTIMIZE HYPERPARAMS
    hyperopt_runs = {}
    trials_dict = {}
    kwargs = {
        'bounds': copy(model.priors['fe_bounds']), 
        'iterations': max_evals,
        'scoring': scoring, 
        'val_size': val_size,
        'min_days': min_days,
    }
    o = Optimiser(model, train, kwargs)

    if num_hyperopt_runs == 1:
        (best_init, n_days), err, trials = o.optimisestar(0)
        hyperopt_runs[err] = (best_init, n_days)
        trials_dict[0] = trials
    else:
        pool = Pool(processes=5)
        for i, ((best_init, n_days), err, trials) in enumerate(pool.map(o.optimisestar, list(range(num_hyperopt_runs)))):
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
    n_days = (test[model.date].max() - train[model.date].min() + timedelta(days=1+predict_days)).days
    predictions.loc[:, model.date] = pd.to_datetime(pd.Series([timedelta(days=x)+train[model.date].min() for x in range(n_days)]))
    predictions.loc[:, model.ycol] = model.predict(train[model.date].min(),
        test[model.date].max() + timedelta(days=predict_days))

    # LOSS
    lc = Loss_Calculator()
    train_pred = predictions[model.ycol][:len(train)]
    trainerr = lc.evaluate(train[model.ycol], train_pred)
    test_pred = predictions[model.ycol][len(train):len(train) + len(test)]
    testerr = lc.evaluate(test[model.ycol], test_pred)
    if xform_func != None:
        xform_trainerr = lc.evaluate(xform_func(train[model.ycol], dtp),
            xform_func(train_pred, dtp))
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
        'mod.params': model.pipeline.mod.params
    }

    return result_dict

def single_cycle(dist, st, area_names, config, model_params):
    dataframes, dtp, model_params, fname = setup(dist, st, area_names, config, model_params)
    create_output_folder(fname)
    xform_func = lograte_to_cumulative if config['log'] else rate_to_cumulative
    return run_cycle(dataframes, model_params, predict_days=config['forecast_days'],
        max_evals=config['max_evals'], num_hyperopt_runs=config['num_hyperopt'], 
        val_size=config['forecast_days'], min_days=config['min_days'], 
        scoring=config['scoring'], dtp=dtp, xform_func=xform_func)