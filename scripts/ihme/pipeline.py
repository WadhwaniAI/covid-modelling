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
from main.ihme.fitting import create_output_folder, single_cycle
from utils.util import read_config

import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', module='pandas', category=RuntimeWarning) #, message='invalid value encountered in')
warnings.filterwarnings('ignore', module='curvefit', category=RuntimeWarning) #, message='invalid value encountered in')

val_size = 7
test_size = 7
min_days = 7
scoring = 'mape'
# -------------------

def run_pipeline(dist, st, area_names, config, model_params):
    start_time = time.time()
    results_dict = single_cycle(dist, st, area_names, model_params, **config)
    output_folder = create_output_folder(f'forecast/{dist}')
    predictions = results_dict['predictions']['predictions']
    train, test, df = results_dict['train'], results_dict['test'], results_dict['df']
    xform_func = lograte_to_cumulative if config['log'] else rate_to_cumulative
    dtp = results_dict['district_total_pop']
    runtime = time.time() - start_time
    print('runtime:', runtime)
    
    # PLOTTING
    plot_df, plot_test = copy(df), copy(test)
    plot_df[model_params['ycol']] = xform_func(plot_df[model_params['ycol']], dtp)
    plot_test[model_params['ycol']] = xform_func(plot_test[model_params['ycol']], dtp)
    predicted_cumulative_deaths = xform_func(predictions[model_params['ycol']], dtp)
    xform_draws = xform_func(results_dict['draws'], dtp)

    plot_results(model_params, results_dict['mod.params'], plot_df, len(train), plot_test, predicted_cumulative_deaths, 
        predictions.index, results_dict['xform_error']['test'], dist, val_size, draws=xform_draws, yaxis_name='cumulative deaths')
    plt.savefig(f'{output_folder}/results.png')
    plt.clf()
    plot_results(model_params, results_dict['mod.params'], df, len(train), test, predictions[model_params['ycol']], 
        predictions.index, results_dict['error']['test'], dist, val_size, draws=results_dict['draws'])
    plt.savefig(f'{output_folder}/results_notransform.png')
    plt.clf()

    # SAVE PARAMS INFO
    with open(f'{output_folder}/params.json', 'w') as pfile:
        pargs = copy(model_params)
        pargs.update(config)
        pargs['func'] = pargs['func'].__name__
        pargs['priors']['fe_init'] = results_dict['fe_init']
        pargs['n_days_train'] = int(results_dict['n_days'])
        pargs['error'] = results_dict['error']
        pargs['runtime'] = runtime
        json.dump(pargs, pfile)

    # SAVE DATA, PREDICTIONS
    picklefn = f'{output_folder}/data.pkl'
    with open(picklefn, 'wb') as pickle_file:
        pickle.dump(results_dict, pickle_file)

    return results_dict
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-d", "--district", help="district name", required=True)
    parser.add_argument("-c", "--config", help="config file name", required=True)
    args = parser.parse_args()
    config, model_params = read_config(args.config)
    dist, st, area_names = cities[args.district]
    run_pipeline(dist, st, area_names, config, model_params)
# -------------------