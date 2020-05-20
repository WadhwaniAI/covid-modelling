import os
import sys
import json
from copy import copy
import random
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sigfig import round
import curvefit
from curvefit.core.functions import *

sys.path.append('../..')
from models.ihme.new_model import IHME
from models.ihme.util import get_mortality, evaluate
from models.ihme.data import get_district_timeseries_cached
from backtesting_hyperparams import optimize_hyperparameters, train_test_split, backtesting
from backtesting_hyperparams import lograte_to_cumulative, rate_to_cumulative
from backtesting_hyperparams import plot_results, plot_backtesting_results
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
        df[smoothedcol] = df[model_params['ycol']].rolling(args.smoothing).mean()
        model_params['ycol'] = smoothedcol
    
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
    # HYPER PARAM TUNING
    if args.search:
        bounds = model.priors['fe_bounds']
        step = (0.1, 2, 0.5)
        n_days_train, fe_init = optimize_hyperparameters(model, train, bounds, step, int(args.search))
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

    # UNCERTAINTY
    draws_dict = model.calc_draws()
    for k in draws_dict.keys():
        low = draws_dict[k]['lower']
        up = draws_dict[k]['upper']
        # TODO: group handling
        draws = np.vstack((low, up))   

    if args.log:
        plot_train, plot_test = copy(train), copy(test)
        plot_train[model_params['ycol']] = lograte_to_cumulative(plot_train[model_params['ycol']], dtp)
        plot_test[model_params['ycol']] = lograte_to_cumulative(plot_test[model_params['ycol']], dtp)
        predicted_cumulative_deaths = lograte_to_cumulative(all_preds, dtp)
        draws = lograte_to_cumulative(draws, dtp)
    else:
        plot_train, plot_test = copy(train), copy(test)
        plot_train[model_params['ycol']] = rate_to_cumulative(plot_train[model_params['ycol']], dtp)
        plot_test[model_params['ycol']] = rate_to_cumulative(plot_test[model_params['ycol']], dtp)
        predicted_cumulative_deaths = rate_to_cumulative(all_preds, dtp)
        draws = rate_to_cumulative(draws, dtp)

    # ADD PLOTTING CODE HERE
    plot_results(model, plot_train, plot_test, predicted_cumulative_deaths, all_preds_dates, testerr, f'new_{file_prefix}', draws=draws)
    plt.savefig(f'{output_folder}/results.png')
    plt.clf()
    # plot_results(model, train, test, all_preds, all_preds_dates, testerr, f'new_{file_prefix}', draws=draws)
    # plt.savefig(f'{output_folder}/results_notransform.png')
    # plt.clf()
    
    # SAVE PARAMS INFO
    with open(f'{output_folder}/params.json', 'w') as pfile:
        pargs = copy(model_params)
        pargs['func'] = pargs['func'].__name__
        del pargs['ycols']
        if args.search:
            pargs['priors']['fe_init'] = fe_init
            pargs['search_iterations'] = args.search
        pargs['error'] = err
        pargs['sd'] = args.sd
        pargs['smoothing'] = args.smoothing
        pargs['log'] = args.log
        json.dump(pargs, pfile)

import pickle
def backtest(triple, args):
    df, dtp, model_params, _, _, model, output_folder, file_prefix = setup(triple, args)
    # df = df[df[model.date] > datetime(year=2020, month=4, day=14)]
    future_days = 7
    search_iterations = int(args.search) if args.search is not None else None
    results = backtesting(model, df, df[model_params['date']].min(), 
        df[model_params['date']].max(), future_days=future_days, 
        hyperopt_val_size=7, optimize=search_iterations)
    # print (results)
    picklefn = f'{output_folder}/backtesting.pkl'
    with open(picklefn, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)
    plot_backtesting_results(model, results['df'], results['results'], results['future_days'], file_prefix, transform_y=lograte_to_cumulative, dtp=dtp)
    
    # picklefn = f'../../main/ihme/output/mortality/Mumbai_deaths/big_run/backtesting.pkl'
    # with open(picklefn, 'rb') as pickle_file:
    #     results = pickle.load(pickle_file)
    # plot_backtesting_results(model, df, results, increment, future_days, file_prefix)
    plt.savefig(f'{output_folder}/backtesting.png')

    # SAVE PARAMS INFO
    with open(f'{output_folder}/params.json', 'w') as pfile:
        pargs = copy(model_params)
        pargs['func'] = pargs['func'].__name__
        del pargs['ycols']
        if args.search:
            pargs['search_iterations'] = args.search
        pargs['sd'] = args.sd
        pargs['smoothing'] = args.smoothing
        pargs['log'] = args.log
        json.dump(pargs, pfile)
    
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-d", "--district", help="district name", required=True)
    parser.add_argument("-l", "--log", help="fit on log", required=False, action='store_true')
    parser.add_argument("-sd", "--sd", help="use social distance covariate", required=False, action='store_true')
    parser.add_argument("-s", "--smoothing", help="how much to smooth, else no smoothing", required=False)
    parser.add_argument("-rs", "--search", help="whether to do randomsearch", required=False)
    parser.add_argument("-b", "--backtest", help="whether to do randomsearch", required=False, action='store_true')
    args = parser.parse_args()

    if args.backtest:
        backtest(cities[args.district], args)
    else:
        run_pipeline(cities[args.district], args)
# -------------------