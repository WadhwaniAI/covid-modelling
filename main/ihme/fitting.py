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

def get_params(label):
    with open('params.json', "r") as paramsfile:
        params = json.load(paramsfile)
        if label not in params:
            print("entry not found in params.json")
            sys.exit(0)
    pargs = params['default']
    pargs.update(params[label])
    return pargs

def get_regional_data(dist, st, area_names, ycol, test_size=7, smooth_window=5, use_tracker=False):
    district_timeseries = get_district_timeseries_cached(dist, st, disable_tracker=not use_tracker)
    df, dtp = get_mortality(district_timeseries, st, area_names)
    
    df.loc[:,'sd'] = df['date'].apply(lambda x: [1.0 if x >= datetime(2020, 3, 24) else 0.0]).tolist()

    smoothedcol = f'{ycol}_smoothed'
    df[smoothedcol] = rollingavg(df[ycol], smooth_window)
    df = df.dropna(subset=[smoothedcol])
    
    startday = df['date'][df['mortality'].gt(1e-15).idxmax()]
    df = df.loc[df['mortality'].gt(1e-15).idxmax():,:]
    df.loc[:, 'day'] = (df['date'] - np.min(df['date'])).apply(lambda x: x.days)
    
    threshold = df['date'].max() - timedelta(days=test_size)
    train, test = train_test_split(df, threshold)
    dataframes = {
        'train': train,
        'test': test,
        'df': df,
    }
    return dataframes, dtp, smoothedcol
    
def setup(dist, st, area_names, label, smooth=False, sd=False, test_size=7, smooth_window=5, use_tracker=False):
    model_params = get_params(label)
    model_params['func'] = getattr(functions, model_params['func'])
    # IHME TODO: move covs to params.json
    model_params['covs'] = ['covs', 'sd', 'covs'] if sd else ['covs', 'covs', 'covs']

    dataframes, dtp, smoothedcol = get_regional_data(dist, st, area_names, model_params["ycol"], smooth_window=smooth_window, test_size=test_size)
    if smooth:
        model_params['ycol'] = smoothedcol
    
    fname = f'{dist}_deaths'
    return dataframes, dtp, model_params, fname

def create_output_folder(fname):
    today = datetime.today()
    output_folder = f'../../outputs/ihme/{fname}/{today}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

def run_cycle(min_days, scoring, val_size, dataframes, model_params, dtp, max_evals, num_hyperopt_runs, xform_func=lograte_to_cumulative):
    model = IHME(model_params)
    train, test = dataframes['train'], dataframes['test']
    fe_init, n_days_train = model.priors['fe_init'], min_days
    hyperopt_runs = {}
    trials_dict = {}
    pool = Pool(processes=5)
    kwargs = {
        'bounds': copy(model.priors['fe_bounds']), 
        'iterations': max_evals,
        'scoring': scoring, 
        'val_size': val_size,
        'min_days': min_days,
    }
    o = Optimiser(model, train, kwargs)
    for i, ((best_init, n_days), err, trials) in enumerate(pool.map(o.optimisestar, list(range(num_hyperopt_runs)))):
        hyperopt_runs[err] = (best_init, n_days)
        trials_dict[i] = trials
    (fe_init, n_days_train) = hyperopt_runs[min(hyperopt_runs.keys())]
    
    train = train[-n_days_train:]
    train.loc[:, 'day'] = (train['date'] - np.min(train['date'])).apply(lambda x: x.days)
    test.loc[:, 'day'] = (test['date'] - np.min(train['date'])).apply(lambda x: x.days)
    model.priors['fe_init'] = fe_init

    model.fit(train)

    n_days = (test[model_params['date']].max() - train[model_params['date']].min() + timedelta(days=1+model_params['daysforward'])).days
    all_preds_dates = pd.to_datetime(pd.Series([timedelta(days=x)+train[model_params['date']].min() for x in range(n_days)]))
    all_preds = model.predict(train[model_params['date']].min(), test[model_params['date']].max() + timedelta(days=model_params['daysforward']))
    
    lc = Loss_Calculator()

    train_pred = all_preds[:len(train)]
    trainerr = lc.evaluate(train[model_params['ycol']], train_pred)
    test_pred = all_preds[len(train):len(train) + len(test)]
    testerr = lc.evaluate(test[model_params['ycol']], test_pred)
    # future_pred = all_preds[len(train) + len(test):]
    err = {
        'train': trainerr,
        "test": testerr
    }

    xform_err = None
    xform_trainerr = lc.evaluate(xform_func(train[model_params['ycol']], dtp),
        xform_func(train_pred, dtp))
    xform_testerr = lc.evaluate(xform_func(test[model_params['ycol']], dtp),
        xform_func(test_pred, dtp))
    xform_err = {
        'train': xform_trainerr,
        "test": xform_testerr
    }

    # UNCERTAINTY
    draws_dict = model.calc_draws()
    for k in draws_dict.keys():
        low = draws_dict[k]['lower']
        up = draws_dict[k]['upper']
        # TODO: group handling
        draws = np.vstack((low, up))   

    error = {
        'xform': xform_err,
        'original': err,
    }
    predictions = pd.DataFrame(columns=[model_params['date'], model_params['ycol']])
    predictions.loc[:,model_params['date']] = all_preds_dates
    predictions.loc[:,model_params['ycol']] = all_preds

    return model, predictions, draws, error, trials_dict
