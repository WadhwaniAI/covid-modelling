import os
import numpy as np
import pandas as pd
import datetime
import argparse

import sys
sys.path.append('../..')
from data.dataloader import get_covid19india_api_data
from main.seir.fitting import get_variable_param_ranges
from recovs import single_fitting_cycle, smooth_using_total
from main.seir.forecast import create_all_csvs, write_csv, plot_forecast
from utils.create_report import create_report

# --- turn into command line args
parser = argparse.ArgumentParser() 
parser.add_argument("-t", "--use-tracker", help="district name", required=False, action='store_true')
parser.add_argument("-i", "--iterations", help="optimiser iterations", required=False, default=700, type=int)
parser.add_argument("-n", "--ndays", help="smoothing days", required=False, default=30, type=int)
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

districts_to_show = [('Maharashtra', 'Mumbai')]

for state, district in districts_to_show:
    predictions_dict[(state, district)] = {}
    predictions_dict[(state, district)]['m1'] = single_fitting_cycle(smooth_using_total,
        dataframes, state, district, train_period=7, val_period=7, 
        n_days_back_smooth=args.ndays,
        data_from_tracker=args.use_tracker, initialisation='intermediate', num_evals=args.iterations,
        which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
    predictions_dict[(state, district)]['m2'] = single_fitting_cycle(smooth_using_total,
        dataframes, state, district, train_period=7, val_period=0, num_evals=args.iterations,
        n_days_back_smooth=args.ndays,
        data_from_tracker=args.use_tracker, initialisation='intermediate',
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
        df_loss_master.loc[region, :] = np.around(predictions_dict[region][m]['df_loss'].values.T.flatten().astype('float'), decimals=2)
        
for region in predictions_dict.keys():
    predictions_dict[region]['forecast'] = plot_forecast(predictions_dict[region], region, both_forecasts=False, error_bars=True)

for region in predictions_dict.keys():
    create_report(predictions_dict[region])

df_output = create_all_csvs(predictions_dict, icu_fraction=0.02)
write_csv(df_output, '../../output-{}.csv'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

