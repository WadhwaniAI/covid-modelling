import numpy as np
import pandas as pd

import datetime
import copy
import json
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from functools import partial

import sys
sys.path.append('../../')

from data.dataloader import get_covid19india_api_data
from data.processing import get_data
from models.ihme.dataloader import get_dataframes_cached

from main.seir.fitting import single_fitting_cycle, get_variable_param_ranges
from main.seir.forecast import get_forecast, create_region_csv, create_all_csvs, write_csv
from main.seir.forecast import order_trials, get_all_trials
from utils.create_report import create_report, trials_to_df
from utils.enums import Columns
from viz import plot_forecast, plot_trials

import pickle
import argparse

parser = argparse.ArgumentParser()
t = time.time()
parser.add_argument("-f", "--folder", help="folder name", required=True, type=str)
parser.add_argument("-i", "--iterations", help="number of trials for beta exploration", required=False, default=1500, type=str)
args = parser.parse_args()   

with open(f'../../reports/{args.folder}/predictions_dict.pkl', 'rb') as pkl:
    region_dict = pickle.load(pkl)

m1_trials = region_dict['m1']['all_trials']

region_dict['m1']['run_params']['val_period']

df_val = region_dict['m1']['df_val']
df_val['hospitalised']

def avg_weighted_error(region_dict, hp):
    beta = hp['beta']
    losses = region_dict['m1']['losses']
    df_val = region_dict['m1']['df_district']         .set_index('date')         .loc[region_dict['m1']['df_val']['date'],:]
    active_predictions = region_dict['m1']['all_trials'].loc[:, df_val.index]
    beta_loss = np.exp(-beta*losses)
    avg_rel_err = 0
    for date in df_val.index:
        weighted_pred = (beta_loss*active_predictions[date]).sum() / beta_loss.sum()
        rel_error = (weighted_pred - df_val.loc[date,'hospitalised']) / df_val.loc[date,'hospitalised']
        avg_rel_err += rel_error
    avg_rel_err /= len(df_val)
    return avg_rel_err

searchspace = {
    'beta': hp.uniform('beta', 0, 10)
}
num_evals = 1000

trials = Trials()
best = fmin(partial(avg_weighted_error, region_dict),
            space=searchspace,
            algo=tpe.suggest,
            max_evals=args.iterations,
            trials=trials)

beta = best['beta']

date_of_interest = '2020-06-15'
date_of_interest = datetime.datetime.strptime(date_of_interest, '%Y-%m-%d')

df_pdf = pd.DataFrame(columns=['loss', 'weight', 'pdf', date_of_interest, 'cdf'])
df_pdf['loss'] = region_dict['m2']['losses']
df_pdf['weight'] = np.exp(-beta*df_pdf['loss'])
df_pdf['pdf'] = df_pdf['weight'] / df_pdf['weight'].sum()
df_pdf[date_of_interest] = region_dict['m2']['all_trials'].loc[:, date_of_interest]

df_pdf[:5]

df_pdf = df_pdf.sort_values(by=date_of_interest)
df_pdf.index.name = 'idx'
df_pdf.reset_index(inplace=True)

df_pdf['cdf'] = df_pdf['pdf'].cumsum()

df_pdf[:5]

percentiles = range(10, 100, 10), np.array([2.5, 5, 95, 97.5])
percentiles = np.sort(np.concatenate(percentiles))
ptile_dict = {}

for ptile in percentiles:
    index_value = (df_pdf['cdf'] - ptile/100).apply(abs).idxmin()
    best_idx = df_pdf.loc[index_value - 2:index_value + 2, :]['idx'].min()
    ptile_dict[ptile] = best_idx

print(ptile_dict)
