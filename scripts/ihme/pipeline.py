import os
import json
from copy import copy
import argparse
import pandas as pd
import numpy as np
import dill as pickle
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import curvefit

import sys
sys.path.append('../..')
from models.ihme.model import IHME
from models.ihme.util import cities
from models.ihme.util import lograte_to_cumulative, rate_to_cumulative

from main.ihme.plotting import plot_results
from main.ihme.fitting import setup, create_output_folder, run_cycle

import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', module='pandas', category=RuntimeWarning) #, message='invalid value encountered in')
warnings.filterwarnings('ignore', module='curvefit', category=RuntimeWarning) #, message='invalid value encountered in')

val_size = 7
test_size = 7
min_days = 7
scoring = 'mape'
# -------------------

def run_pipeline(dist, st, area_names, args):
    label = 'log_mortality' if args.log else 'mortality'
    
    start_time = time.time()
    dataframes, dtp, model_params, file_prefix = setup(dist, st, area_names, label)
    output_folder = create_output_folder(f'{file_prefix}')
    
    xform_func = lograte_to_cumulative if args.log else rate_to_cumulative
    train, test, df = dataframes['train'], dataframes['test'], dataframes['df']
    model, predictions, draws, error, trials_dict = run_cycle(min_days, scoring, val_size, dataframes, model_params, dtp, args.max_evals, args.hyperopt, xform_func=xform_func)
    runtime = time.time() - start_time
    print('runtime:', runtime)
    
    # PLOTTING
    plot_df, plot_test = copy(df), copy(test)
    plot_df[model_params['ycol']] = xform_func(plot_df[model_params['ycol']], dtp)
    plot_test[model_params['ycol']] = xform_func(plot_test[model_params['ycol']], dtp)
    predicted_cumulative_deaths = xform_func(predictions[model_params['ycol']], dtp)
    xform_draws = xform_func(draws, dtp)

    plot_results(model, plot_df, len(train), plot_test, predicted_cumulative_deaths, 
        predictions[model_params['date']], error['xform']['test'], f'new_{file_prefix}', val_size, draws=xform_draws, yaxis_name='cumulative deaths')
    plt.savefig(f'{output_folder}/results.png')
    plt.clf()
    plot_results(model, df, len(train), test, predictions[model_params['ycol']], 
        predictions[model_params['date']], error['original']['test'], f'new_{file_prefix}', val_size, draws=draws)
    plt.savefig(f'{output_folder}/results_notransform.png')
    plt.clf()

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
        pargs['priors']['fe_init'] = [int(i) for i in model.priors['fe_init']]
        pargs['n_days_train'] = len(model.pipeline.all_data)
        pargs['error'] = error
        pargs['runtime'] = runtime
        json.dump(pargs, pfile)

    # SAVE DATA, PREDICTIONS
    picklefn = f'{output_folder}/data.pkl'
    with open(picklefn, 'wb') as pickle_file:
        data = {
            'data': df,
            'train': train,
            'test': test,
            'dates': predictions[model_params['date']],
            'predictions': predictions[model_params['ycol']],
            'cumulative_predictions': predicted_cumulative_deaths,
            'trials': trials_dict
        }
        pickle.dump(data, pickle_file)

# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-d", "--district", help="district name", required=True)
    parser.add_argument("-l", "--log", help="fit on log", required=False, action='store_true')
    parser.add_argument("-sd", "--sd", help="use social distance covariate", required=False, action='store_true')
    parser.add_argument("-s", "--smoothing", help="how much to smooth, else no smoothing", required=False, type=int)
    parser.add_argument("-hp", "--hyperopt", help="[single run only] number of times to do hyperparam optimization", required=False, type=int, default=1)
    parser.add_argument("-i", "--max_evals", help="max evals on each hyperopt run", required=False, default=50, type=int)
    parser.add_argument("-dt", "--disable_tracker", help="disable tracker (use athena instead)", required=False, action='store_true')
    args = parser.parse_args()

    dist, st, area_names = cities[args.district]
    run_pipeline(dist, st, area_names, args)
# -------------------