import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import geoplot as gplt
import geoplot.crs as gcrs
import seaborn as sns
import matplotlib.dates as mdates

from scipy.stats import zscore

import os
import copy
import pickle
import re
from datetime import datetime, date, timedelta
from glob import glob

from utils.generic.config import read_config, make_date_key_str
from utils.generic.reichlab import *
from viz.reichlab import *

us_states_abbv_df = pd.read_csv('../../data/data/us_states_abbv.csv')
us_states_abbv_dict = dict(zip(us_states_abbv_df['state'], us_states_abbv_df['state_code']))

predictions_pkl_filename = '/scratch/users/sansiddh/covid-modelling/2020_1102_210541/predictions_dict.pkl'
print('Reading pkl file of run 2020_1102_210541')
with open(predictions_pkl_filename, 'rb') as f:
    predictions_dict_1 = pickle.load(f)

predictions_pkl_filename = '/scratch/users/sansiddh/covid-modelling/2020_1103_013041/predictions_dict.pkl'
print('Reading pkl file of run 2020_1103_013041')
with open(predictions_pkl_filename, 'rb') as f:
    predictions_dict_2 = pickle.load(f)

predictions_pkl_filename = '/scratch/users/sansiddh/covid-modelling/2020_1103_043227/predictions_dict.pkl'
print('Reading pkl file of run 2020_1103_043227')
with open(predictions_pkl_filename, 'rb') as f:
    predictions_dict_3 = pickle.load(f)

predictions_dict_comb = {}
for loc, loc_dict in predictions_dict_1.items():
    predictions_dict_comb[loc] = {}
    predictions_dict_comb[loc]['m1'] = {}
    predictions_dict_comb[loc]['m1']['df_district'] = copy.deepcopy(
        predictions_dict_1[loc]['m1']['df_district']
    )
    predictions_dict_comb[loc]['m1']['df_train'] = copy.deepcopy(
        predictions_dict_1[loc]['m1']['df_train']
    )
    predictions_dict_comb[loc]['m1']['df_val'] = copy.deepcopy(
        predictions_dict_1[loc]['m1']['df_val']
    )
    predictions_dict_comb[loc]['m1']['trials_processed'] = {}
    predictions_dict_comb[loc]['m1']['trials_processed']['predictions'] = \
        predictions_dict_1[loc]['m1']['trials_processed']['predictions'] + \
        predictions_dict_2[loc]['m1']['trials_processed']['predictions'] + \
        predictions_dict_3[loc]['m1']['trials_processed']['predictions']
    
    predictions_dict_comb[loc]['m1']['trials_processed']['losses'] = np.concatenate(
        (predictions_dict_1[loc]['m1']['trials_processed']['losses'],  
         predictions_dict_2[loc]['m1']['trials_processed']['losses'], 
         predictions_dict_3[loc]['m1']['trials_processed']['losses']), 
        axis=None
    )
    
    predictions_dict_comb[loc]['m1']['trials_processed']['params'] = np.concatenate(
        (predictions_dict_1[loc]['m1']['trials_processed']['params'],  
         predictions_dict_2[loc]['m1']['trials_processed']['params'], 
         predictions_dict_3[loc]['m1']['trials_processed']['params']), 
        axis=None
    )

config_filename = 'us2.yaml'
config = read_config(config_filename)

wandb_config = read_config(config_filename, preprocess=False)
wandb_config = make_date_key_str(wandb_config)

output_folder = '/scratch/users/{}/covid-modelling/{}'.format(
    'sansiddh', 'aug22_combined')
os.makedirs(output_folder, exist_ok=True)

for i, (loc, loc_dict) in enumerate(predictions_dict_comb.items()):
    uncertainty_args = {'predictions_dict': loc_dict, 'fitting_config': config['fitting'],
                        'forecast_config': config['forecast'], 'process_trials': False,
                        **config['uncertainty']['uncertainty_params']}
    print(f'Fitting beta for {loc}...')
    uncertainty = config['uncertainty']['method'](**uncertainty_args)

    loc_dict['m2'] = {}
    loc_dict['m2']['forecasts'] = {}
    uncertainty_forecasts = uncertainty.get_forecasts()
    for key in uncertainty_forecasts.keys():
        loc_dict['m2']['forecasts'][key] = uncertainty_forecasts[key]['df_prediction']

    loc_dict['m2']['forecasts']['ensemble_mean'] = uncertainty.ensemble_mean_forecast
    print(f'Fitting done for {loc}. {i+1}/{len(predictions_dict_comb)} done.')
    with open(f'{output_folder}/predictions_dict.pkl', 'wb') as f:
        pickle.dump(predictions_dict_comb, f)
    print(f'Saved predictions_dict.')
        
