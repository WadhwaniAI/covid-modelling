
import os
from datetime import datetime
import argparse
import pickle
import json
import matplotlib.pyplot as plt

import sys
sys.path.append('../..')
from data.processing import get_dataframes_cached
from main.seir.fitting import get_regional_data
from main.seir.backtesting import SEIRBacktest
from utils.generic.config import read_config
from viz import plot_backtest, plot_backtest_errors

districts_dict = {
    'pune': ('Maharashtra', 'Pune'), 
    'mumbai': ('Maharashtra', 'Mumbai'), 
    'jaipur': ('Rajasthan', 'Jaipur'), 
    'ahmedabad': ('Gujarat', 'Ahmedabad'), 
    'bengaluru': ('Karnataka', 'Bengaluru Urban'),
    'delhi': ('Delhi', None)
}

parser = argparse.ArgumentParser()
now = datetime.now().strftime("%Y%m%d-%H%M%S")
parser.add_argument("-d", "--district", help="district name", required=True, type=str)
parser.add_argument("-c", "--config", help="path to config file", required=True, type=str)
parser.add_argument("-f", "--folder", help="folder name", required=False, default=None, type=str)
parser.add_argument("-r", "--replot", help="folder name", required=False, default=None, type=str)
args = parser.parse_args()
if args.folder is None:
    folder = f'backtesting/{args.district.lower()}/{str(now)}'
config, model_params = read_config(args.config, backtesting=True)
print('\rgetting dataframes...')
dataframes = get_dataframes_cached()

state, district = districts_dict[args.district]
fit = config['fit'] # 'm1' or 'm2'
data_from_tracker=not config['disable_tracker']

output_folder = f'../../outputs/seir/{folder}'
if not os.path.exists(output_folder):
        os.makedirs(output_folder)

if args.replot is None:
    print('\rgetting district data...')
    df_district, df_district_raw_data = get_regional_data(dataframes, state, district, 
        data_from_tracker=data_from_tracker, filename=None, data_format=None,
        smooth_jump=config['smooth_jump'], smoothing_method=config['smooth_method'], 
        smoothing_length=config['smooth_ndays'])

    print('starting backtesting...')
    backtester = SEIRBacktest(state, district, df_district, df_district_raw_data, data_from_tracker)
    results = backtester.test(fit, increment=config['increment'], num_evals=config['max_evals'],
            train_period=config['min_days'], val_period=config['val_size'],
            future_days=config['forecast_days'])
    picklefn = f'{output_folder}/backtesting.pkl'

    with open(picklefn, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)
else:
    with open(f'{output_folder}/backtesting.pkl', 'rb') as pfile:
        results = pickle.load(pfile)

    with open(f'{output_folder}/params.json', 'r') as pfile:
        config = json.load(pfile)

    print('\rgetting district data...')
    df_district, df_district_raw_data = get_regional_data(dataframes, state, district, 
        data_from_tracker=config['data_from_tracker'], filename=None, data_format=None,
        smooth_jump=config['smooth_jump'], smoothing_method=config['smooth_method'], 
        smoothing_length=config['smooth_ndays'])

plt.clf()
plot_backtest(results['results'], df_district, district, savepath=f'{output_folder}/backtest.png')

plt.clf()
plot_backtest_errors(results['results'], df_district, district, savepath=f'{output_folder}/errors-backtest.png')

params = {
    'state': state,
    'district': district,
    'fit': fit,
    'data_from_tracker': data_from_tracker,
}
with open(f'{output_folder}/params.json', 'w') as pfile:
        params.update(config)
        json.dump(params, pfile)
print(f'see output at: {output_folder}')
