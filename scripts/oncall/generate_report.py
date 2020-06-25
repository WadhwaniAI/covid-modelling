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
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
parser.add_argument("-d", "--district", help="district name", required=True, type=str)
parser.add_argument("-c", "--config", help="path to config file", required=True, type=str)
parser.add_argument("-f", "--folder", help="folder name", required=False, default=str(now), type=str)
args = parser.parse_args()

config, model_params = read_config(args.config)
print(config.keys())
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
state, district = districts_dict[args.district.strip().lower()]

predictions_dict['m1'] = single_fitting_cycle(
    dataframes, state, district, train_period=7, val_period=7, num_evals=config['max_evals'],
    data_from_tracker=not config['disable_tracker'], initialisation='intermediate', model=SEIR_Testing, 
    # filename='../../data/data/mumbai_2020_06_02.csv', data_format='new',
    smooth_jump=config['smooth_jump'], smoothing_method=config['smooth_method'], smoothing_length=config['smooth_ndays'],
    # which_compartments=['deceased', 'total_infected'])
    which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
predictions_dict['m2'] = single_fitting_cycle(
    dataframes, state, district, train_period=7, val_period=0, num_evals=config['max_evals'],
    data_from_tracker=not config['disable_tracker'], initialisation='intermediate', model=SEIR_Testing, 
    # filename='../../data/data/mumbai_2020_06_02.csv', data_format='new',
    smooth_jump=config['smooth_jump'], smoothing_method=config['smooth_method'], smoothing_length=config['smooth_ndays'],
    # which_compartments=['deceased', 'total_infected'])
    which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
    
predictions_dict['state'] = state
predictions_dict['dist'] = district
predictions_dict['fitting_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
predictions_dict['datasource'] = 'covid19api' if predictions_dict['m1']['data_from_tracker'] else 'municipality'
predictions_dict['variable_param_ranges'] = predictions_dict['m1']['variable_param_ranges']
predictions_dict['data_last_date'] = predictions_dict['m2']['data_last_date']

predictions_dict['m2']['forecast'] = plot_forecast(predictions_dict, (state, district), both_forecasts=False, error_bars=True, days=config['forecast_days'])
    
predictions, losses, params = get_all_trials(predictions_dict, train_fit='m1', forecast_days=config['forecast_days'])
predictions_dict['m1']['params'] = params
predictions_dict['m1']['losses'] = losses
predictions_dict['m1']['predictions'] = predictions
predictions_dict['m1']['all_trials'] = trials_to_df(predictions, losses, params, column=Columns.active)
predictions, losses, params = get_all_trials(predictions_dict, train_fit='m2', forecast_days=config['forecast_days'])
predictions_dict['m2']['params'] = params
predictions_dict['m2']['losses'] = losses
predictions_dict['m2']['predictions'] = predictions
predictions_dict['m2']['all_trials'] = trials_to_df(predictions, losses, params, column=Columns.active)
kforecasts = plot_trials(
    predictions_dict,
    train_fit='m2',
    predictions=predictions, 
    losses=losses, params=params, 
    k=config['ktrials'],
    which_compartments=[Columns.confirmed, Columns.active])
predictions_dict['m2']['forecast_confirmed_topk'] = kforecasts[Columns.confirmed]
predictions_dict['m2']['forecast_active_topk'] = kforecasts[Columns.active]

create_report(predictions_dict, ROOT_DIR=f'../../reports/{args.folder}') 
predictions_dict['m1']['all_trials'].to_csv(f'../../reports/{args.folder}/m1-trials.csv')
predictions_dict['m2']['all_trials'].to_csv(f'../../reports/{args.folder}/m2-trials.csv')
predictions_dict['m2']['df_district_unsmoothed'].to_csv(f'../../reports/{args.folder}/true.csv')
predictions_dict['m2']['df_district'].to_csv(f'../../reports/{args.folder}/smoothed.csv')

df_output = create_region_csv(predictions_dict, region=district, regionType='district', icu_fraction=0.02, days=config['forecast_days'])
write_csv(df_output, filename=f'../../reports/{args.folder}/output-{now}.csv')

print(f"yeet. done: view files at ../../reports/{args.folder}/")
