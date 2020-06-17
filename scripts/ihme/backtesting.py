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
from pathos.multiprocessing import ProcessingPool as Pool

import curvefit
from curvefit.core import functions

sys.path.append('../..')
from models.ihme.model import IHME
from models.ihme.util import get_mortality
from models.ihme.util import lograte_to_cumulative, rate_to_cumulative
from models.ihme.population import get_district_population
from models.ihme.util import cities

from data.processing import get_district_timeseries_cached

from main.ihme.backtesting import IHMEBacktest
from main.ihme.optimiser import Optimiser
from main.ihme.plotting import plot_results, plot_backtesting_results, plot_backtesting_errors
from main.ihme.plotting import plot
from main.ihme.fitting import setup, create_output_folder

from utils.util import train_test_split, rollingavg
from utils.loss import Loss_Calculator

import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', module='pandas', category=RuntimeWarning) #, message='invalid value encountered in')
warnings.filterwarnings('ignore', module='curvefit', category=RuntimeWarning) #, message='invalid value encountered in')

increment = 3
future_days = 7
val_size = 7
test_size = 7
min_days = 7
scoring = 'mape'
# -------------------

def backtest(dist, st, area_names, args):
    label = 'log_mortality' if args.log else 'mortality'
    dataframes, dtp, model_params, file_prefix = setup(dist, st, area_names, label)
    output_folder = create_output_folder(f'backtesting/{file_prefix}')
    df = dataframes['df']
    
    start_time = time.time()
    # df = df[df[model.date] > datetime(year=2020, month=4, day=14)]
    xform = lograte_to_cumulative if args.log else rate_to_cumulative
    model = IHME(model_params)
    backtester = IHMEBacktest(model, df, dist, st)
    results = backtester.test(future_days=future_days, 
        hyperopt_val_size=val_size,
        max_evals=args.max_evals, increment=increment, xform_func=xform,
        dtp=dtp, min_days=min_days)
    picklefn = f'{output_folder}/backtesting.pkl'
    with open(picklefn, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)
            
    backtester.plot_results(file_prefix, scoring=scoring, transform_y=xform, dtp=dtp, axis_name='cumulative deaths', savepath=f'{output_folder}/backtesting.png') 
    backtester.plot_errors(file_prefix, scoring='mape', use_xform=True, savepath=f'{output_folder}/backtesting_mape.png') 
    backtester.plot_errors(file_prefix, scoring='rmse', use_xform=True, savepath=f'{output_folder}/backtesting_rmse.png') 
    backtester.plot_errors(file_prefix, scoring='rmsle', use_xform=True, savepath=f'{output_folder}/backtesting_rmsle.png') 

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

def replot_backtest(dist, st, area_names, folder, args):
    dtp = get_district_population(st, area_names)
    file_prefix = f'{dist}_deaths'
    root_folder = create_output_folder(f'backtesting/{file_prefix}/{folder}')
    output_folder = os.path.join(root_folder, '/replotted/')
    start_time = time.time()

    xform = lograte_to_cumulative if args.log else rate_to_cumulative
            
    picklefn = f'{root_folder}/backtesting.pkl'
    with open(picklefn, 'rb') as pickle_file:
        results = pickle.load(pickle_file)
    model = results['model']
    df = results['df']
    
    backtester = IHMEBacktest(model, df, dist, st)
    
    backtester.plot_results(file_prefix, results=results['results'], scoring=scoring, transform_y=xform, dtp=dtp, axis_name='cumulative deaths', savepath=f'{output_folder}/backtesting.png') 
    backtester.plot_errors(file_prefix, results=results['results'], scoring='mape', use_xform=True, savepath=f'{output_folder}/backtesting_mape.png') 
    backtester.plot_errors(file_prefix, results=results['results'], scoring='rmse', use_xform=True, savepath=f'{output_folder}/backtesting_rmse.png') 
    backtester.plot_errors(file_prefix, results=results['results'], scoring='rmsle', use_xform=True, savepath=f'{output_folder}/backtesting_rmsle.png') 

    dates = pd.Series(list(results['results'].keys())).apply(lambda x: results['df']['date'].min() + timedelta(days=x))
    plot(dates, [d['n_days'] for d in results['results'].values()], 'n_days_train', 'n_days')
    plt.savefig(f'{output_folder}/backtesting_ndays.png')
    plt.clf()

    runtime = time.time() - start_time
    print('time:', runtime)

# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-d", "--district", help="district name", required=True)
    parser.add_argument("-l", "--log", help="fit on log", required=False, action='store_true')
    parser.add_argument("-sd", "--sd", help="use social distance covariate", required=False, action='store_true')
    parser.add_argument("-s", "--smoothing", help="how much to smooth, else no smoothing", required=False, type=int)
    parser.add_argument("-hp", "--hyperopt", help="[single run only] number of times to do hyperparam optimization", required=False, type=int, default=1)
    parser.add_argument("-i", "--max_evals", help="max evals on each hyperopt run", required=False, default=50, type=int)
    parser.add_argument("-re", "--replot", help="folder of backtest run to replot, must also run with -b", required=False)
    parser.add_argument("-dt", "--disable_tracker", help="disable tracker (use athena instead)", required=False, action='store_true')
    args = parser.parse_args()
    dist, st, area_names = cities[args.district]

    if args.replot:
        replot_backtest(dist, st, area_names, args.replot, args)
    else:
        backtest(dist, st, area_names, args)
# -------------------