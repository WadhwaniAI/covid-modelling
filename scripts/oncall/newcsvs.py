import numpy as np
import pandas as pd

import datetime
import copy
import json
import time
import argparse
import pickle
import os

import sys
sys.path.append('../../')

from main.seir.forecast import get_forecast, create_region_csv, get_all_trials
from main.seir.forecast import create_decile_csv
from main.seir.fitting import calculate_loss, train_val_split
from utils.enums import Columns
from viz import plot_r0_multipliers
from utils.create_report import trials_to_df
# -----

parser = argparse.ArgumentParser()
t = time.time()
parser.add_argument("-f", "--folder", help="folder name", required=True, type=str)
parser.add_argument("-d", "--district", help="district name", required=True, type=str)
args = parser.parse_args()   

with open(f'../../reports/{args.folder}/predictions_dict.pkl', 'rb') as pkl:
    region_dict = pickle.load(pkl)

# predictions, losses, params = get_all_trials(region_dict, train_fit='m1')
# region_dict['m1']['params'] = params
# region_dict['m1']['losses'] = losses
# region_dict['m1']['predictions'] = predictions
# region_dict['m1']['all_trials'] = trials_to_df(predictions, losses, params)
# predictions, losses, params = get_all_trials(region_dict, train_fit='m2')
# region_dict['m2']['params'] = params
# region_dict['m2']['losses'] = losses
# region_dict['m2']['predictions'] = predictions
# region_dict['m2']['all_trials'] = trials_to_df(predictions, losses, params)

# with open(f'../../reports/{args.folder}/predictions_dict.pkl', 'wb') as pkl:
#     pickle.dump(region_dict, pkl)

pune_deciles_idx = {
    2.5: 28,
    5: 13,
    10: 1,
    20: 10,
    30: 0,
    40: 5,
    50: 2,
    60: 4,
    70: 3,
    80: 7,
    90: 11,
    95: 8,
    97.5: 55,
}
mum_deciles_idx = {
    2.5: 174,
    5: 27,
    10: 22,
    20: 1,
    30: 53,
    40: 3,
    50: 10,
    60: 157,
    70: 30,
    80: 2,
    90: 42,
    95: 78,
    97.5: 164,
}

if args.district == 'mumbai':
    deciles_idx = mum_deciles_idx
elif args.district == 'pune':
    deciles_idx = pune_deciles_idx
else:
    raise Exception("invalid district")


from main.seir.fitting import get_regional_data
from data.processing import get_dataframes_cached

# first one for mumbai, second for pune, third if pkl was produced after june 6
# df_reported, _ = get_regional_data(get_dataframes_cached(), 'Maharashtra', 'Mumbai', False, None, None)
# df_reported = region_dict['m2']['df_district'] # change this to df_district_unsmoothed
df_reported = region_dict['m2']['df_district_unsmoothed'] # change this to df_district_unsmoothed

params = region_dict['m2']['params']

from main.seir.uncertainty import MCUncertainty
uncertainty = MCUncertainty(region_dict, None)
deciles_forecast = uncertainty.get_forecasts(ptile_dict=deciles_idx)
deciles_params = {key: val['params'] for key, val in deciles_forecast.items()}

# deciles to csv
df_output = create_decile_csv(deciles_forecast, df_reported, region_dict['dist'], 'district', icu_fraction=0.02)
df_output.to_csv(f'../../reports/{args.folder}/deciles.csv')
with open(f'../../reports/{args.folder}/deciles-params.json', 'w+') as params_json:
    json.dump(deciles_params, params_json)

# TODO: combine these two
df_reported.to_csv(f'../../reports/{args.folder}/true.csv')
region_dict['m2']['df_district'].to_csv(f'../../reports/{args.folder}/smoothed.csv')

from main.seir.forecast import predict_r0_multipliers, save_r0_mul

# perform what-ifs on 80th percentile
predictions_mul_dict = predict_r0_multipliers(region_dict, params[deciles_idx[80]])
save_r0_mul(predictions_mul_dict, folder=args.folder)

ax = plot_r0_multipliers(region_dict, params[deciles_idx[80]], predictions_mul_dict, multipliers=[0.9, 1, 1.1, 1.25])
ax.figure.savefig(f'../../reports/{args.folder}/what-ifs/what-ifs.png')

print(f"yuh. done: view files at ../../reports/{args.folder}/")
