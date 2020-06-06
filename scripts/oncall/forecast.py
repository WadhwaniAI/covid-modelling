import numpy as np
import pandas as pd

import datetime
import copy
import json
import time

import sys
sys.path.append('../../')

from data.dataloader import get_covid19india_api_data
from data.processing import get_data
from models.ihme.dataloader import get_dataframes_cached

from main.seir.fitting import single_fitting_cycle, get_variable_param_ranges
from main.seir.fitting import calculate_loss, train_val_split
from main.seir.forecast import get_forecast, create_decile_csv, write_csv
from main.seir.forecast import order_trials, get_all_trials
from utils.create_report import create_report, trials_to_df
from utils.enums import Columns
from viz import plot_forecast, plot_trials, plot_r0_multipliers
from main.seir.uncertainty import get_all_ptiles, forecast_ptiles
from main.seir.uncertainty import save_r0_mul, predict_r0_multipliers

import pickle
import argparse

parser = argparse.ArgumentParser()
t = time.time()
parser.add_argument("-f", "--folder", help="folder name", required=True, type=str)
parser.add_argument("-d", "--district", help="district name", required=True, type=str)
parser.add_argument("-i", "--iterations", help="number of trials for beta exploration", required=False, default=1500, type=str)
args = parser.parse_args()   

with open(f'../../reports/{args.folder}/predictions_dict.pkl', 'rb') as pkl:
    region_dict = pickle.load(pkl)

if args.district == 'pune':
    date_of_interest = '2020-06-15'
elif args.district == 'mumbai':
    date_of_interest = '2020-06-30'
else:
    raise Exception("unknown district")

deciles_idx = get_all_ptiles(region_dict, date_of_interest, args.iterations)

from main.seir.fitting import get_regional_data
from models.ihme.dataloader import get_dataframes_cached

# first one for mumbai, second for pune, third if pkl was produced after june 6
# df_reported, _ = get_regional_data(get_dataframes_cached(), 'Maharashtra', 'Mumbai', False, None, None)
# df_reported = region_dict['m2']['df_district'] # change this to df_district_unsmoothed
df_reported = region_dict['m2']['df_district_unsmoothed'] # change this to df_district_unsmoothed

# forecast deciles to csv
deciles_params, deciles_forecast = forecast_ptiles(region_dict, deciles_idx)
df_output = create_decile_csv(deciles_forecast, df_reported, region_dict['dist'], 'district', icu_fraction=0.02)
df_output.to_csv(f'../../reports/{args.folder}/deciles.csv')
with open(f'../../reports/{args.folder}/deciles-params.json', 'w+') as params_json:
    json.dump(deciles_params, params_json)

# TODO: combine these two
# df_reported.to_csv(f'../../reports/{args.folder}/true.csv')
# region_dict['m2']['df_district'].to_csv(f'../../reports/{args.folder}/smoothed.csv')

# perform what-ifs on 80th percentile
params = region_dict['m2']['params']
predictions_mul_dict = predict_r0_multipliers(region_dict, params[deciles_idx[80]])
save_r0_mul(predictions_mul_dict, folder=args.folder)
with open(f'../../reports/{args.folder}/what-ifs/what-ifs-params.json', 'w+') as params_json:
    json.dump({key: val['lockdown_R0'] for key, val in predictions_mul_dict.items()}, params_json)

ax = plot_r0_multipliers(region_dict, params[deciles_idx[80]], predictions_mul_dict, multipliers=[0.9, 1, 1.1, 1.25])
ax.figure.savefig(f'../../reports/{args.folder}/what-ifs/what-ifs.png')
