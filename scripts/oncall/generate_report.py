import numpy as np
import pandas as pd

import datetime
import copy
import json
import time
import argparse

import sys
sys.path.append('../../')

from data.processing import get_data, get_dataframes_cached

from models.seir import SEIR_Testing, SEIRHD, SEIR_Movement, SEIR_Movement_Testing

from main.seir.fitting import single_fitting_cycle, get_variable_param_ranges
from main.seir.forecast import get_forecast, create_region_csv, write_csv
from main.seir.forecast import order_trials, get_all_trials
from viz import plot_forecast, plot_trials
from utils.create_report import create_report, trials_to_df
from utils.enums import Columns
from utils.util import read_config

'''
Please keep this script at par functionally with 
    notebooks/seir/[STABLE] generate_report.ipynb
AND linearly runnable 
Use command line args / Modify the parameters in `single_fitting_cycle` and `plot_trials` per customizations

ex. 
python3 generate_report.py --districts mumbai,pune --ktrials 10 -i 700 -f reporttest
python3 generate_report.py --districts mumbai --ktrials 100 -i 1000 -f reporttest -s -n 33
'''

# --- turn into command line args
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--use-tracker", help="whether to use covid19api tracker", required=False, action='store_true')
parser.add_argument("-s", "--smooth-jump", help="smooth jump", required=False, action='store_true')
parser.add_argument("-method", "--smooth-method", help="smooth method", required=False, default='weighted', type=str)
parser.add_argument("-i", "--iterations", help="optimiser iterations", required=False, default=700, type=int)
parser.add_argument("-n", "--ndays", help="smoothing days", required=False, default=33, type=int)
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
parser.add_argument("-f", "--folder", help="folder name", required=False, default=str(now), type=str)
parser.add_argument("-k", "--ktrials", help="k trials to forecast", required=False, default=10, type=int)
parser.add_argument("-d", "--districts", help="districts", required=True, type=str)
parser.add_argument("--fdays",help="how many days to forecast for", required=False, default=30, type=int)
args = parser.parse_args()

dataframes = get_dataframes_cached()

predictions_dict = {}

districts_dict = {
    'pune': ('Maharashtra', 'Pune'), 
    'mumbai': ('Maharashtra', 'Mumbai'), 
    'jaipur': ('Rajasthan', 'Jaipur'), 
    'ahmedabad': ('Gujarat', 'Ahmedabad'), 
    'bengaluru': ('Karnataka', 'Bengaluru Urban'),
    'delhi': ('Delhi', None)
}
state, district = districts_dict[args.districts.strip().lower()]

predictions_dict['m1'] = single_fitting_cycle(
    dataframes, state, district, train_period=7, val_period=7, num_evals=args.iterations,
    data_from_tracker=args.use_tracker, initialisation='intermediate', model=SEIR_Testing, 
    # filename='../../data/data/mumbai_2020_06_02.csv', data_format='new',
    smooth_jump=args.smooth_jump, smoothing_method=args.smooth_method, smoothing_length=args.ndays,
    # which_compartments=['deceased', 'total_infected'])
    which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
predictions_dict['m2'] = single_fitting_cycle(
    dataframes, state, district, train_period=7, val_period=0, num_evals=args.iterations,
    data_from_tracker=args.use_tracker, initialisation='intermediate', model=SEIR_Testing, 
    # filename='../../data/data/mumbai_2020_06_02.csv', data_format='new',
    smooth_jump=args.smooth_jump, smoothing_method=args.smooth_method, smoothing_length=args.ndays,
    # which_compartments=['deceased', 'total_infected'])
    which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
    
predictions_dict['state'] = state
predictions_dict['dist'] = district
predictions_dict['fitting_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
predictions_dict['datasource'] = 'covid19api' if predictions_dict['m1']['data_from_tracker'] else 'municipality'
predictions_dict['variable_param_ranges'] = predictions_dict['m1']['variable_param_ranges']
predictions_dict['data_last_date'] = predictions_dict['m2']['data_last_date']

predictions_dict['m2']['forecast'] = plot_forecast(predictions_dict, (state, district), both_forecasts=False, error_bars=True, days=args.fdays)
    
predictions, losses, params = get_all_trials(predictions_dict, train_fit='m1', forecast_days=args.fdays)
predictions_dict['m1']['params'] = params
predictions_dict['m1']['losses'] = losses
predictions_dict['m1']['predictions'] = predictions
predictions_dict['m1']['all_trials'] = trials_to_df(predictions, losses, params)
predictions, losses, params = get_all_trials(predictions_dict, train_fit='m2', forecast_days=args.fdays)
predictions_dict['m2']['params'] = params
predictions_dict['m2']['losses'] = losses
predictions_dict['m2']['predictions'] = predictions
predictions_dict['m2']['all_trials'] = trials_to_df(predictions, losses, params)
kforecasts = plot_trials(
    predictions_dict,
    train_fit='m2',
    predictions=predictions, 
    losses=losses, params=params, 
    k=args.ktrials,
    which_compartments=[Columns.confirmed, Columns.active])
predictions_dict['m2']['forecast_confirmed_topk'] = kforecasts[Columns.confirmed]
predictions_dict['m2']['forecast_active_topk'] = kforecasts[Columns.active]

create_report(predictions_dict, ROOT_DIR=f'../../reports/{args.folder}') 
predictions_dict['m1']['all_trials'].to_csv(f'../../reports/{args.folder}/m1-trials.csv')
predictions_dict['m2']['all_trials'].to_csv(f'../../reports/{args.folder}/m2-trials.csv')
predictions_dict['m2']['df_district_unsmoothed'].to_csv(f'../../reports/{args.folder}/true.csv')
predictions_dict['m2']['df_district'].to_csv(f'../../reports/{args.folder}/smoothed.csv')

df_output = create_region_csv(predictions_dict, region=district, regionType='district', icu_fraction=0.02, days=args.fdays)
write_csv(df_output, filename=f'../../reports/{args.folder}/output-{now}.csv')

print(f"yeet. done: view files at ../../reports/{args.folder}/")
