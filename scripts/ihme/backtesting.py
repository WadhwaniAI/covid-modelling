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
from viz import plot_backtest, plot_backtest_errors
from utils.enums import Columns

from utils.util import read_config

import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', module='pandas', category=RuntimeWarning) #, message='invalid value encountered in')
warnings.filterwarnings('ignore', module='curvefit', category=RuntimeWarning) #, message='invalid value encountered in')

# -------------------

def backtest(dist, st, area_names, config, model_params, folder):
    dataframes, dtp, model_params = setup(dist, st, area_names, model_params, **config)
    output_folder = create_output_folder(f'backtesting/{folder}')
    df = dataframes['df']
    
    start_time = time.time()
    # df = df[df[model.date] > datetime(year=2020, month=4, day=14)]
    model = IHME(model_params)
    which_compartments = Columns.which_compartments()
    backtester = IHMEBacktest(model, df, dist, st)
    xform_func = lograte_to_cumulative if config['log'] else rate_to_cumulative
    res = backtester.test(future_days=config['forecast_days'], 
        hyperopt_val_size=config['val_size'],
        max_evals=config['max_evals'], increment=config['increment'], xform_func=xform_func,
        dtp=dtp, min_days=config['min_days'], which_compartments=which_compartments)
    picklefn = f'{output_folder}/backtesting.pkl'
    with open(picklefn, 'wb') as pickle_file:
            pickle.dump(res, pickle_file)
    
    plot_backtest(
        res['results'], res['df'], dist, which_compartments=which_compartments,
        scoring=config['scoring'], axis_name='cumulative deaths', savepath=f'{output_folder}/backtesting.png') 
    
    plot_backtest_errors(
        res['results'], res['df'], dist, which_compartments=which_compartments,
        scoring=config['scoring'], savepath='{fldr}/backtesting_{scoring}.png'.format(fldr=output_folder, scoring=config['scoring'])) 

    # dates = pd.Series(list(res['results'].keys())).apply(lambda x: res['df']['date'].min() + timedelta(days=x))
    # plot(dates, [d['n_days'] for d in res['results'].values()], 'n_days_train', 'n_days', savepath=f'{output_folder}/backtesting_ndays.png')

    runtime = time.time() - start_time
    print('time:', runtime)

    # SAVE PARAMS INFO
    with open(f'{output_folder}/params.json', 'w') as pfile:
        pargs = copy(model_params)
        pargs.update(config)
        pargs['func'] = pargs['func'].__name__
        pargs['runtime'] = runtime
        json.dump(pargs, pfile)
    print(f"yee we done see results here: {output_folder}")

def replot_backtest(dist, st, area_names, folder):
    dtp = get_district_population(st, area_names)
    file_prefix = f'{dist}_deaths'
    output_folder = create_output_folder(f'backtesting/{folder}/replotted')
    root_folder = os.path.dirname(output_folder)
    start_time = time.time()

    paramsjson = f'{root_folder}/params.json'
    with open(paramsjson, 'r') as paramsfile:
        config = json.load(paramsfile)

    picklefn = f'{root_folder}/backtesting.pkl'
    with open(picklefn, 'rb') as pickle_file:
        results = pickle.load(pickle_file)

        plot_backtest(
            results['results'], results['df'], config['ycol'], file_prefix, 
            scoring=config['scoring'], dtp=dtp, axis_name='cumulative deaths', savepath=f'{output_folder}/backtesting.png') 
        plot_backtest_errors(
            results['results'], results['df'], config['ycol'], file_prefix, 
            scoring=config['scoring'], savepath='{fldr}/backtesting_{scoring}.png'.format(fldr=output_folder, scoring=config['scoring'])) 

        # dates = pd.Series(list(results['results'].keys())).apply(lambda x: results['df']['date'].min() + timedelta(days=x))
        # plot(dates, [d['n_days'] for d in results['results'].values()], 'n_days_train', 'n_days', savepath=f'{output_folder}/backtesting_ndays.png')

    runtime = time.time() - start_time
    print('time:', runtime)
    print(f"yee we done see results here: {output_folder}")

# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-d", "--district", help="district name", required=True)
    parser.add_argument("-c", "--config", help="config file name", required=True)
    parser.add_argument("-re", "--replot", help="folder of backtest run to replot", required=False)
    parser.add_argument("-f", "--folder", help="folder name (to save in or to replot from)", required=False, default=None, type=str)
    args = parser.parse_args()
    config, model_params = read_config(args.config, backtesting=True)
    dist, st, area_names = cities[args.district]

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = f'{args.district}/{str(now)}' if args.folder is None else args.folder

    if args.replot:
        replot_backtest(dist, st, area_names, args.replot)
    else:
        backtest(dist, st, area_names, config, model_params, folder)
# -------------------