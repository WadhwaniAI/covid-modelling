import os
import numpy as np
import pandas as pd
import datetime
import argparse

import sys
sys.path.append('../..')
from data.dataloader import get_covid19india_api_data
from main.seir.fitting import single_fitting_cycle, get_variable_param_ranges
from main.seir.forecast import create_all_csvs, write_csv, plot_forecast
from utils.create_report import create_report

# --- turn into command line args
parser = argparse.ArgumentParser() 
parser.add_argument("-t", "--use-tracker", help="district name", required=False, action='store_true')
args = parser.parse_args()
use_tracker = args.use_tracker
# ---

dataframes = get_covid19india_api_data()
predictions_dict = {}
save_json = {
    'variable_param_ranges': get_variable_param_ranges(as_str=True)
}
model_report = {}
today = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
output_folder = os.path.join(os.getcwd(), f'output/{today}')
if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
# districts_to_show = [('Maharashtra', 'Pune'), 
#                      ('Maharashtra', 'Mumbai'), 
#                      ('Rajasthan', 'Jaipur'), 
#                      ('Gujarat', 'Ahmedabad'), 
#                      ('Karnataka', 'Bengaluru Urban'),
#                      ('Delhi', None)]

districts_to_show = [('Maharashtra', 'Pune'), ('Maharashtra', 'Mumbai')]

for state, district in districts_to_show:
    predictions_dict[(state, district)] = {}
    predictions_dict[(state, district)]['m1'] = single_fitting_cycle(
        dataframes, state, district, train_period=7, val_period=7, 
        data_from_tracker=args.use_tracker, initialisation='intermediate', num_evals=700,
        which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
    predictions_dict[(state, district)]['m2'] = single_fitting_cycle(
        dataframes, state, district, train_period=7, val_period=0, num_evals=700,
        train_on_val=True, data_from_tracker=args.use_tracker, initialisation='intermediate',
        which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
    
    predictions_dict[(state, district)]['state'] = state
    predictions_dict[(state, district)]['dist'] = district
    predictions_dict[(state, district)]['fitting_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
    predictions_dict[(state, district)]['datasource'] = 'covid19api' if predictions_dict[(state, district)]['m1']['data_from_tracker'] else 'municipality'
    predictions_dict[(state, district)]['variable_param_ranges'] = predictions_dict[(state, district)]['m1']['variable_param_ranges']
    predictions_dict[(state, district)]['data_last_date'] = predictions_dict[(state, district)]['m2']['data_last_date']

for m in ['m1', 'm2']:
    starting_key = list(predictions_dict.keys())[0]

    loss_columns = pd.MultiIndex.from_product([predictions_dict[starting_key][m]['df_loss'].columns, predictions_dict[starting_key][m]['df_loss'].index])
    loss_index = predictions_dict.keys()

    df_loss_master = pd.DataFrame(columns=loss_columns, index=loss_index)
    for region in predictions_dict.keys():
        # region_folder = os.path.join(output_folder, region[1])
        # if not os.path.exists(region_folder):
        #     os.makedirs(region_folder)

        # report = model_report[region]
        df_loss_master.loc[region, :] = np.around(predictions_dict[region][m]['df_loss'].values.T.flatten().astype('float'), decimals=2)
        
        # print (df_loss_master)
        # report[m] = {}
        # report[m]['loss_df'] = df_loss_master.to_markdown()
        # report[m]['best_params'] = predictions_dict[region][m]['best_params']
        # report[m]['plot'] = predictions_dict[region][m]['ax'].figure
        # report[m]['plot_path'] = f'{m}-fit.png'
        # report[m]['plot'].savefig(os.path.join(region_folder, report[m]['plot_path']))

for region in predictions_dict.keys():
    predictions_dict[region]['forecast'] = plot_forecast(predictions_dict[region], region, both_forecasts=False, error_bars=True)

for region in predictions_dict.keys():
    create_report(predictions_dict[region])

# for region in predictions_dict.keys():
#     region_folder = os.path.join(output_folder, region[1])
#     if not os.path.exists(region_folder):
#         os.makedirs(region_folder)
#     report = model_report[region]['overall']
    
#     m2_val_m1 = plot_forecast(predictions_dict[region], region, both_forecasts=False, error_bars=True,)
    
#     report['m2_val_m1'] = m2_val_m1.figure
#     report['m2_val_m1_path'] = f'm2_val_m1-fit.png'
#     report['m2_val_m1'].savefig(os.path.join(region_folder, report['m2_val_m1_path']))
    
#     m2_trials = predictions_dict[region]['m2']['trials']
#     m2_best_trials = sorted(m2_trials.trials, key=lambda x: x['result']['loss'], reverse=False)
#     report['top_10_trials'] = [trial['misc']['vals'] for trial in m2_best_trials[:10]]
    
#     create_report(path=region_folder, **model_report[region])

df_output = create_all_csvs(predictions_dict, initialisation='intermediate', train_period=7, icu_fraction=0.02)
write_csv(df_output, '../../output-{}.csv'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

