import os
import sys
import json
from copy import copy
import random
import argparse
import pandas as pd
import numpy as np
import dill as pickle
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import curvefit
from curvefit.core.functions import *

from pathos.multiprocessing import ProcessingPool as Pool

sys.path.append('../..')
from models.ihme.new_model import IHME
from models.ihme.util import get_mortality
from models.ihme.dataloader import get_district_timeseries_cached
from backtesting_hyperparams import Optimize, backtesting
from utils.util import train_test_split, rollingavg
from utils.loss import evaluate
from models.ihme.util import lograte_to_cumulative, rate_to_cumulative
from plotting import plot_results, plot_backtesting_results, plot_backtesting_errors
from plotting import plot
pd.options.mode.chained_assignment = None


# tuples: (district, state, census_area_name(s))
mumbai = 'Mumbai', 'Maharashtra', ['District - Mumbai (23)', 'District - Mumbai Suburban (22)']
amd = 'Ahmedabad', 'Gujarat', ['District - Ahmadabad (07)']
jaipur = 'Jaipur', 'Rajasthan', ['District - Jaipur (12)']
pune = 'Pune', 'Maharashtra', ['District - Pune (25)']
delhi = 'Delhi', 'Delhi', ['State - NCT OF DELHI (07)']
bengaluru = 'Bengaluru', 'Karnataka', ['District - Bangalore (18)', 'District - Bangalore Rural (29)']

cities = {
    'mumbai': mumbai,
    'ahmedabad': amd,
    'jaipur': jaipur,
    'pune': pune,
    'delhi': delhi,
    'bengaluru': bengaluru,
}

increment = 3
future_days = 7
val_size = 7
min_days = 7

# -------------------
def setup(triple, args):
    dist, st, area_names = triple
    fname = f'{dist}_deaths'

    district_timeseries = get_district_timeseries_cached(dist, st)
    df, dtp = get_mortality(district_timeseries, st, area_names)

    # TODO: move these flags to params, make label cl flag?
    label = f'log_mortality' if args.log else 'mortality'

    with open('params.json', "r") as paramsfile:
        params = json.load(paramsfile)
        if label not in params:
            print("entry not found in params.json")
            sys.exit(0)
    pargs = params['default']
    pargs.update(params[label])
    model_params = pargs

    # set vars

    model_params['ycol'] = f'log_mortality' if args.log else 'mortality'
    model_params['func'] = log_erf if args.log else erf

    df['date']= pd.to_datetime(df['date'])
    df.loc[:,'group'] = len(df) * [ 1.0 ]
    df.loc[:,'covs'] = len(df) * [ 1.0 ]
    df.loc[:,'sd'] = df['date'].apply(lambda x: [1.0 if x >= datetime(2020, 3, 24) else 0.0]).tolist()

    if args.smoothing:
        print(f'smoothing {args.smoothing}')
        smoothedcol = f'{model_params["ycol"]}_smoothed'
        df[smoothedcol] = rollingavg(df[model_params['ycol']], args.smoothing)
        model_params['ycol'] = smoothedcol
        df = df.dropna(subset=[smoothedcol])
    
    covs = ['covs', 'sd', 'covs'] if args.sd else ['covs', 'covs', 'covs']
    model_params['covs'] = covs
    
    startday = df['date'][df['mortality'].gt(1e-15).idxmax()]
    df = df.loc[df['mortality'].gt(1e-15).idxmax():,:]
    df.loc[:, 'day'] = (df['date'] - np.min(df['date'])).apply(lambda x: x.days)
    
    test_size = 7
    threshold = df['date'].max() - timedelta(days=test_size)
    train, test = train_test_split(df, threshold)
    
    model = IHME(model_params)

    # output
    today = datetime.today()
    if args.backtest:
        output_folder = f'output/backtesting/{fname}/{today}'
    else:
        output_folder = f'output/mortality/{fname}/{today}'
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    return df, dtp, model_params, train, test, model, output_folder, fname

def run_pipeline(triple, args):
    df, dtp, model_params, train, test, model, output_folder, file_prefix = setup(triple, args)
    start_time = time.time()
    # HYPER PARAM TUNING
    if args.hyperopt:
        bounds = model.priors['fe_bounds']
        
        hyperopt_runs = {}
        trials_dict = {}
        pool = Pool(processes=5)
        o = Optimize((model, train,
                bounds, args.max_evals, 'mape', 
                val_size, min_days))
        for i, ((best_init, n_days), err, trials) in enumerate(pool.map(o.optimizestar, list(range(args.hyperopt)))):
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
    
    train_pred = all_preds[:len(train)]
    trainerr = evaluate(train[model_params['ycol']], train_pred)
    test_pred = all_preds[len(train):len(train) + len(test)]
    testerr = evaluate(test[model_params['ycol']], test_pred)
    # future_pred = all_preds[len(train) + len(test):]
    err = {
        'train': trainerr,
        "test": testerr
    }

    xform_err = None
    xform_func = lograte_to_cumulative if args.log else rate_to_cumulative
    xform_trainerr = evaluate(xform_func(train[model_params['ycol']], dtp),
        xform_func(train_pred, dtp))
    xform_testerr = evaluate(xform_func(test[model_params['ycol']], dtp),
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

    plot_df, plot_test = copy(df), copy(test)
    plot_df[model_params['ycol']] = xform_func(plot_df[model_params['ycol']], dtp)
    plot_test[model_params['ycol']] = xform_func(plot_test[model_params['ycol']], dtp)
    predicted_cumulative_deaths = xform_func(all_preds, dtp)
    xform_draws = xform_func(draws, dtp)

    # ADD PLOTTING CODE HERE
    plot_results(model, plot_df, len(train), plot_test, predicted_cumulative_deaths, 
        all_preds_dates, xform_testerr, f'new_{file_prefix}', val_size, draws=xform_draws, yaxis_name='cumulative deaths')
    plt.savefig(f'{output_folder}/results.png')
    plt.clf()
    plot_results(model, df, len(train), test, all_preds, 
        all_preds_dates, testerr, f'new_{file_prefix}', val_size, draws=draws)
    plt.savefig(f'{output_folder}/results_notransform.png')
    plt.clf()
    
    runtime = time.time() - start_time
    print('time:', runtime)

    # SAVE PARAMS INFO
    with open(f'{output_folder}/params.json', 'w') as pfile:
        pargs = copy(model_params)
        pargs['func'] = pargs['func'].__name__
        del pargs['ycols']
        pargs['hyperopt'] = args.hyperopt
        pargs['max_evals'] = args.max_evals
        pargs['sd'] = args.sd
        pargs['smoothing'] = args.smoothing
        pargs['log'] = args.log
        pargs['priors']['fe_init'] = [int(i) for i in fe_init]
        pargs['n_days_train'] = int(n_days_train)
        pargs['error'] = err
        pargs['xform_error'] = xform_err
        pargs['runtime'] = runtime
        json.dump(pargs, pfile)

    # SAVE DATA, PREDICTIONS
    picklefn = f'{output_folder}/data.pkl'
    with open(picklefn, 'wb') as pickle_file:
        data = {
            'data': df,
            'train': train,
            'test': test,
            'dates': all_preds_dates,
            'predictions': all_preds,
            'cumulative_predictions': predicted_cumulative_deaths,
            'trials': trials_dict
        }
        pickle.dump(data, pickle_file)
    
def backtest(triple, args):
    df, dtp, model_params, _, _, model, output_folder, file_prefix = setup(triple, args)
    start_time = time.time()
    # df = df[df[model.date] > datetime(year=2020, month=4, day=14)]
    xform = lograte_to_cumulative if args.log else rate_to_cumulative
    results = backtesting(model, df, df[model_params['date']].min(), 
        df[model_params['date']].max(), future_days=future_days, 
        hyperopt_val_size=val_size, optimize_runs=args.hyperopt,
        max_evals=args.max_evals, increment=increment, xform_func=xform,
        dtp=dtp, min_days=min_days)
    # print (results)
    picklefn = f'{output_folder}/backtesting.pkl'
    with open(picklefn, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)
    plot_backtesting_results(model, results['df'], results['results'],
        results['future_days'], file_prefix, transform_y=xform, dtp=dtp,
            axis_name='cumulative deaths')    
    # picklefn = f'../../main/ihme/output/mortality/Mumbai_deaths/big_run/backtesting.pkl'
    # with open(picklefn, 'rb') as pickle_file:
    #     results = pickle.load(pickle_file)
    # plot_backtesting_results(model, df, results, increment, future_days, file_prefix)
    plt.savefig(f'{output_folder}/backtesting.png')
    plt.clf()
    plot_backtesting_errors(model, df, df[model_params['date']].min(),
        results['results'], file_prefix, scoring='mape', use_xform=True)
    plt.savefig(f'{output_folder}/backtesting_mape.png')
    plt.clf()
    plot_backtesting_errors(model, df, df[model_params['date']].min(),
        results['results'], file_prefix, scoring='rmse', use_xform=True)
    plt.savefig(f'{output_folder}/backtesting_rmse.png')
    plt.clf()
    plot_backtesting_errors(model, df, df[model_params['date']].min(),
        results['results'], file_prefix, scoring='rmsle', use_xform=True)
    plt.savefig(f'{output_folder}/backtesting_rmsle.png')
    plt.clf()

    dates = pd.Series(list(results['results'].keys())).apply(lambda x: results['df']['date'].min() + timedelta(days=x))
    plot(dates, [d['n_days'] for d in results['results'].values()], 'n_days_train', 'n_days')
    plt.savefig(f'{output_folder}/backtesting_ndays.png')
    plt.clf()

    runtime = time.time() - start_time
    print('time:', runtime)

    # SAVE PARAMS INFO
    with open(f'{output_folder}/params.json', 'w') as pfile:
        pargs = copy(model_params)
        pargs['func'] = pargs['func'].__name__
        del pargs['ycols']
        pargs['hyperopt'] = args.hyperopt
        pargs['max_evals'] = args.max_evals
        pargs['sd'] = args.sd
        pargs['smoothing'] = args.smoothing
        pargs['log'] = args.log
        pargs['increment'] = increment
        pargs['future_days'] = future_days
        pargs['min_days'] = min_days
        pargs['hyperopt_val_size'] = val_size
        pargs['runtime'] = runtime
        json.dump(pargs, pfile)

def replot_backtest(triple, folder, args):
    dist, _, _ = triple
    output_folder = folder + '/replotted'
    file_prefix = f'{dist}_deaths'
    
    _, dtp, _, _, _, model, _, _ = setup(triple, args)
    with open(f'{output_folder}/params.json', 'r') as pfile:
        pargs = json.load(pfile)
        args.hyperopt = pargs['hyperopt']
        args.max_evals = pargs['max_evals']
        args.sd = pargs['sd']
        args.smoothing = pargs['smoothing']
        args.log = pargs['log']
        increment = pargs['increment']
        future_days = pargs['future_days']
        min_days = pargs['min_days']
        val_size = pargs['hyperopt_val_size']
        # increment = 3
        # future_days = 7
        # val_size = 7
        # min_days = 7
    # df = df[df[model.date] > datetime(year=2020, month=4, day=14)]
    
    xform = lograte_to_cumulative if args.log else rate_to_cumulative 
    
    picklefn = f'../../main/ihme/output/mortality/Mumbai_deaths/big_run/backtesting.pkl'
    with open(picklefn, 'rb') as pickle_file:
        pkl = pickle.load(pickle_file)
        results = pkl['results']
        df = pkl['df']
        # dtp = pkl['dtp']
        # model = pkl['model']

    plot_backtesting_results(model, results['df'], results['results'],
        results['future_days'], file_prefix, transform_y=xform, dtp=dtp,
            axis_name='cumulative deaths') 
    # plot_backtesting_results(model, df, results, increment, future_days, file_prefix)
    plt.savefig(f'{output_folder}/replotted/backtesting.png')
    plt.clf()
    plot_backtesting_errors(model, df, df[model.date].min(),
        results['results'], file_prefix, scoring='mape', use_xform=True)
    plt.savefig(f'{output_folder}/replotted/backtesting_mape.png')
    plt.clf()
    plot_backtesting_errors(model, df, df[model.date].min(),
        results['results'], file_prefix, scoring='rmse', use_xform=True)
    plt.savefig(f'{output_folder}/replotted/backtesting_rmse.png')
    plt.clf()
    plot_backtesting_errors(model, df, df[model.date].min(),
        results['results'], file_prefix, scoring='rmsle', use_xform=True)
    plt.savefig(f'{output_folder}/replotted/backtesting_rmsle.png')
    plt.clf()

# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-d", "--district", help="district name", required=True)
    parser.add_argument("-l", "--log", help="fit on log", required=False, action='store_true')
    parser.add_argument("-sd", "--sd", help="use social distance covariate", required=False, action='store_true')
    parser.add_argument("-s", "--smoothing", help="how much to smooth, else no smoothing", required=False, type=int)
    parser.add_argument("-hp", "--hyperopt", help="number of times to do hyperparam optimization", required=False, type=int, default=0)
    parser.add_argument("-i", "--max_evals", help="max evals on each hyperopt run", required=False, default=50, type=int)
    parser.add_argument("-b", "--backtest", help="run backtesting", required=False, action='store_true')
    args = parser.parse_args()

    if args.backtest:
        backtest(cities[args.district], args)
    else:
        run_pipeline(cities[args.district], args)
# -------------------