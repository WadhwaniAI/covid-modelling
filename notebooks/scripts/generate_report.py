import numpy as np
import pandas as pd

import datetime
import copy
import json
import time
import argparse

import sys
sys.path.append('../../')

from data.dataloader import get_covid19india_api_data
from data.processing import get_data
from models.ihme.dataloader import get_dataframes_cached

from models.seir.seir_testing import SEIR_Testing
from models.seir.seirhd import SEIRHD
from models.seir.seir_movement import SEIR_Movement
from models.seir.seir_movement_testing import SEIR_Movement_Testing

from main.seir.fitting import single_fitting_cycle, get_variable_param_ranges
from main.seir.forecast import get_forecast, create_region_csv, create_all_csvs, write_csv, plot_forecast
from main.seir.forecast import order_trials, plot_trials
from utils.create_report import create_report
from utils.enums import Columns

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
parser.add_argument("-t", "--use-tracker", help="district name", required=False, action='store_true')
parser.add_argument("-s", "--smooth-jump", help="smooth jump", required=False, action='store_true')
parser.add_argument("-method", "--smooth-method", help="smooth method", required=False, default='weighted', type=str)
parser.add_argument("-i", "--iterations", help="optimiser iterations", required=False, default=700, type=int)
parser.add_argument("-n", "--ndays", help="smoothing days", required=False, default=33, type=int)
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
parser.add_argument("-f", "--folder", help="folder name", required=False, default=str(now), type=str)
parser.add_argument("-k", "--ktrials", help="k trials to forecast", required=False, default=10, type=int)
parser.add_argument("-d", "--districts", help="districts", required=False, default='mumbai', type=str)
args = parser.parse_args()

# dataframes = get_covid19india_api_data()
dataframes = get_dataframes_cached()

predictions_dict = {}

distlist = args.districts.strip().lower()
if ',' in distlist:
    distlist =[d.strip() for d in distlist.split(',')]
else:
    distlist = [distlist]

districts_dict = {
    'pune': ('Maharashtra', 'Pune'), 
    'mumbai': ('Maharashtra', 'Mumbai'), 
    'jaipur': ('Rajasthan', 'Jaipur'), 
    'ahmedabad': ('Gujarat', 'Ahmedabad'), 
    'bengaluru': ('Karnataka', 'Bengaluru Urban'),
    'delhi': ('Delhi', None)
}

districts_to_show = [districts_dict[key] for key in distlist]

for state, district in districts_to_show:
    predictions_dict[(state, district)] = {}
    predictions_dict[(state, district)]['m1'] = single_fitting_cycle(
        dataframes, state, district, train_period=7, val_period=7, num_evals=args.iterations,
        data_from_tracker=args.use_tracker, initialisation='intermediate', model=SEIR_Testing, 
        # filename='../../data/data/mumbai_2020_06_02.csv', data_format='new',
        smooth_jump=args.smooth_jump, smoothing_method=args.smooth_method, smoothing_length=args.ndays,
        # which_compartments=['hospitalised', 'total_infected'])
        which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
    predictions_dict[(state, district)]['m2'] = single_fitting_cycle(
        dataframes, state, district, train_period=7, val_period=0, num_evals=args.iterations,
        data_from_tracker=args.use_tracker, initialisation='intermediate', model=SEIR_Testing, 
        # filename='../../data/data/mumbai_2020_06_02.csv', data_format='new',
        smooth_jump=args.smooth_jump, smoothing_method=args.smooth_method, smoothing_length=args.ndays,
        # which_compartments=['hospitalised', 'total_infected'])
        which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
    
    predictions_dict[(state, district)]['state'] = state
    predictions_dict[(state, district)]['dist'] = district
    predictions_dict[(state, district)]['fitting_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
    predictions_dict[(state, district)]['datasource'] = 'covid19api' if predictions_dict[(state, district)]['m1']['data_from_tracker'] else 'municipality'
    predictions_dict[(state, district)]['variable_param_ranges'] = predictions_dict[(state, district)]['m1']['variable_param_ranges']
    predictions_dict[(state, district)]['data_last_date'] = predictions_dict[(state, district)]['m2']['data_last_date']

for train_fit in ['m1', 'm2']:
    starting_key = list(predictions_dict.keys())[0]

    loss_columns = pd.MultiIndex.from_product([predictions_dict[starting_key][train_fit]['df_loss'].columns, predictions_dict[starting_key][train_fit]['df_loss'].index])
    loss_index = predictions_dict.keys()

    df_loss_master = pd.DataFrame(columns=loss_columns, index=loss_index)
    for key in predictions_dict.keys():
        df_loss_master.loc[key, :] = np.around(predictions_dict[key][train_fit]['df_loss'].values.T.flatten().astype('float'), decimals=2)

for region in predictions_dict.keys():
    predictions_dict[region]['forecast'] = {}
    predictions_dict[region]['forecast']['forecast'] = plot_forecast(predictions_dict[region], region, both_forecasts=False, error_bars=True)
    
    params_array, losses_array = order_trials(predictions_dict[region]['m2'])
    predictions_dict[region]['forecast']['params'] = params_array
    predictions_dict[region]['forecast']['losses'] = losses_array
    kforecasts = plot_trials(predictions_dict[region], which_compartments=[Columns.confirmed, Columns.active], k=args.ktrials)
    predictions_dict[region]['forecast']['forecast_confirmed_topk'] = kforecasts[Columns.confirmed]
    predictions_dict[region]['forecast']['forecast_active_topk'] = kforecasts[Columns.active]

for region in predictions_dict.keys():
    create_report(predictions_dict[region], ROOT_DIR=f'../../reports/{args.folder}') 

df_output = create_all_csvs(predictions_dict, icu_fraction=0.02)
write_csv(df_output, filename=f'../../reports/{args.folder}/output-{now}.csv')