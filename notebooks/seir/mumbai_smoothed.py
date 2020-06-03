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

from models.seir.seir_testing import SEIR_Testing
from models.seir.seirhd import SEIRHD
from models.seir.seir_movement import SEIR_Movement
from models.seir.seir_movement_testing import SEIR_Movement_Testing

from main.seir.fitting import single_fitting_cycle, get_variable_param_ranges
from main.seir.forecast import get_forecast, create_region_csv, create_all_csvs, write_csv, plot_forecast
from utils.create_report import create_report

# dataframes = get_covid19india_api_data()
dataframes = get_dataframes_cached()
t = time.time()

predictions_dict = {}

# districts_to_show = [('Maharashtra', 'Pune'), 
#                      ('Maharashtra', 'Mumbai'), 
#                      ('Rajasthan', 'Jaipur'), 
#                      ('Gujarat', 'Ahmedabad'), 
#                      ('Karnataka', 'Bengaluru Urban'),
#                      ('Delhi', None)]

districts_to_show = [('Maharashtra', 'Mumbai')]

for state, district in districts_to_show:
    predictions_dict[(state, district)] = {}
    predictions_dict[(state, district)]['m1'] = single_fitting_cycle(
        dataframes, state, district, train_period=7, val_period=7, 
        data_from_tracker=False, initialisation='intermediate', num_evals=700, model=SEIR_Testing, 
        smooth_jump=True, smoothing_method='linear', smoothing_length=45,
        which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
    predictions_dict[(state, district)]['m2'] = single_fitting_cycle(
        dataframes, state, district, train_period=7, val_period=0, num_evals=700,
        data_from_tracker=False, initialisation='intermediate', model=SEIR_Testing, 
        smooth_jump=True, smoothing_method='linear', smoothing_length=45,
        which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
    
    predictions_dict[(state, district)]['state'] = state
    predictions_dict[(state, district)]['dist'] = district
    predictions_dict[(state, district)]['fitting_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
    predictions_dict[(state, district)]['datasource'] = 'covid19api' if predictions_dict[(state, district)]['m1']['data_from_tracker'] else 'municipality'
    predictions_dict[(state, district)]['variable_param_ranges'] = predictions_dict[(state, district)]['m1']['variable_param_ranges']
    predictions_dict[(state, district)]['data_last_date'] = predictions_dict[(state, district)]['m2']['data_last_date']

    starting_key = list(predictions_dict.keys())[0]

loss_columns = pd.MultiIndex.from_product([predictions_dict[starting_key]['m1']['df_loss'].columns, predictions_dict[starting_key]['m1']['df_loss'].index])
loss_index = predictions_dict.keys()

df_loss_master = pd.DataFrame(columns=loss_columns, index=loss_index)
for key in predictions_dict.keys():
    df_loss_master.loc[key, :] = np.around(predictions_dict[key]['m1']['df_loss'].values.T.flatten().astype('float'), decimals=2)
    
df_loss_master

starting_key = list(predictions_dict.keys())[0]

loss_columns = pd.MultiIndex.from_product([predictions_dict[starting_key]['m2']['df_loss'].columns, predictions_dict[starting_key]['m2']['df_loss'].index])
loss_index = predictions_dict.keys()

df_loss_master = pd.DataFrame(columns=loss_columns, index=loss_index)
for key in predictions_dict.keys():
    df_loss_master.loc[key, :] = np.around(predictions_dict[key]['m2']['df_loss'].values.T.flatten().astype('float'), decimals=2)
    
df_loss_master

for region in predictions_dict.keys():
    predictions_dict[region]['forecast'] = plot_forecast(predictions_dict[region], region, both_forecasts=False, error_bars=True)

for region in predictions_dict.keys():
    create_report(predictions_dict[region], ROOT_DIR=f'../../reports/{t}') 

df_output = create_all_csvs(predictions_dict, icu_fraction=0.02)
write_csv(df_output)

m = 'm2'

params_array = []
for trial in predictions_dict[(state, district)][m]['trials']:
    params_dict = copy.copy(trial['misc']['vals'])
    for key in params_dict.keys():
        params_dict[key] = params_dict[key][0]
    params_array.append(params_dict)
params_array = np.array(params_array)

losses_array = np.array([trial['result']['loss'] for trial in predictions_dict[(state, district)][m]['trials']])
least_losses_indices = np.argsort(losses_array)
top10losses = losses_array[least_losses_indices][:10]

top10params = params_array[least_losses_indices[:10]]

top10predictions = [get_forecast(predictions_dict[('Maharashtra', 'Mumbai')],
                                  best_params=params_dict) for params_dict in top10params]


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

def plot_trials(m_dict, top10losses, top10params, top10predictions):
    df_true = m_dict['df_district']
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(df_true['date'], df_true['total_infected'],
            '-o', color='C0', label='Confirmed Cases (Observed)')
    for i, df_prediction in enumerate(top10predictions):
        loss_value = np.around(np.sort(top10losses)[:10][i], 2)
        sns.lineplot(x="date", y="total_infected", data=df_prediction,
                    ls='-', label='Confirmed Cases ({})'.format(loss_value))
        plt.text(x=df_prediction['date'].iloc[-1], y=df_prediction['total_infected'].iloc[-1], s=loss_value)
        
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.ylabel('No of People', fontsize=16)
    plt.yscale('log')
    plt.xlabel('Time', fontsize=16)
    plt.legend()
    plt.title('Forecast - ({} {})'.format(region[0], region[1]), fontsize=16)
    #plt.grid()
    return ax

for region in predictions_dict.keys():
    predictions_dict[region][m]['top10params'] = top10params
    predictions_dict[region][m]['forecast_top10'] = plot_trials(predictions_dict[region][m], top10losses, top10params, top10predictions)

for region in predictions_dict.keys():
    create_report(predictions_dict[region], ROOT_DIR=f'../../reports/{t}')                        