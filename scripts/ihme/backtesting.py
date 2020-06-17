import os
import sys
import json
from copy import copy
import argparse
import pandas as pd
import dill as pickle
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

sys.path.append('../..')
from models.ihme.model import IHME
from models.ihme.util import lograte_to_cumulative, rate_to_cumulative
from models.ihme.population import get_district_population
from models.ihme.util import cities

from main.ihme.backtesting import IHMEBacktest
from main.ihme.plotting import plot
from main.ihme.fitting import setup, create_output_folder

from utils.util import read_config

import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', module='pandas', category=RuntimeWarning) #, message='invalid value encountered in')
warnings.filterwarnings('ignore', module='curvefit', category=RuntimeWarning) #, message='invalid value encountered in')

# -------------------

def backtest(dist, st, area_names, config, model_params):
    dataframes, dtp, model_params, file_prefix = setup(dist, st, area_names, model_params, **config)
    output_folder = create_output_folder(f'backtesting/{file_prefix}')
    df = dataframes['df']
    
    start_time = time.time()
    # df = df[df[model.date] > datetime(year=2020, month=4, day=14)]
    xform = lograte_to_cumulative if config['log'] else rate_to_cumulative
    model = IHME(model_params)
    backtester = IHMEBacktest(model, df, dist, st)
    results = backtester.test(future_days=config['forecast_days'], 
        hyperopt_val_size=config['val_size'],
        max_evals=config['max_evals'], increment=config['increment'], xform_func=xform,
        dtp=dtp, min_days=config['min_days'])
    picklefn = f'{output_folder}/backtesting.pkl'
    with open(picklefn, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)
            
    backtester.plot_results(file_prefix, scoring=config['scoring'], transform_y=xform, dtp=dtp, axis_name='cumulative deaths', savepath=f'{output_folder}/backtesting.png') 
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
        pargs.update(config)
        pargs['func'] = pargs['func'].__name__
        pargs['runtime'] = runtime
        json.dump(pargs, pfile)

def replot_backtest(dist, st, area_names, folder):
    dtp = get_district_population(st, area_names)
    file_prefix = f'{dist}_deaths'
    root_folder = create_output_folder(f'backtesting/{file_prefix}/{folder}')
    output_folder = os.path.join(root_folder, '/replotted/')
    start_time = time.time()

    paramsjson = f'{root_folder}/params.json'
    with open(paramsjson, 'r') as paramsfile:
        config = json.load(paramsfile)

    xform = lograte_to_cumulative if config['log'] else rate_to_cumulative
            
    picklefn = f'{root_folder}/backtesting.pkl'
    with open(picklefn, 'rb') as pickle_file:
        results = pickle.load(pickle_file)
    model = results['model']
    df = results['df']
    
    backtester = IHMEBacktest(model, df, dist, st)
    
    backtester.plot_results(file_prefix, results=results['results'], scoring=config['scoring'], transform_y=xform, dtp=dtp, axis_name='cumulative deaths', savepath=f'{output_folder}/backtesting.png') 
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
    parser.add_argument("-c", "--config", help="config file name", required=True)
    parser.add_argument("-re", "--replot", help="folder of backtest run to replot, must also run with -b", required=False)
    args = parser.parse_args()
    config, model_params = read_config(args.config, backtesting=True)
    dist, st, area_names = cities[args.district]

    if args.replot:
        replot_backtest(dist, st, area_names, args.replot)
    else:
        backtest(dist, st, area_names, config, model_params)
# -------------------