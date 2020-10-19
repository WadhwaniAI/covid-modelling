import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import copy
import time
import os
import pickle
import argparse

import sys
sys.path.append('../../')

from data.processing import get_data

import models

from main.seir.fitting import single_fitting_cycle
from main.seir.forecast import get_forecast, forecast_all_trials, create_all_trials_csv, create_decile_csv_new
from main.seir.sensitivity import calculate_sensitivity_and_plot
from utils.generic.create_report import save_dict_and_create_report
from utils.generic.config import read_config, process_config, make_date_key_str
from utils.generic.enums import Columns
from utils.fitting.loss import Loss_Calculator
from utils.generic.logging import log_wandb
from viz import plot_forecast, plot_top_k_trials, plot_ptiles

import yaml
import wandb

def run_single_config_end_to_end(config, wandb_config, run_name):
    predictions_dict = {}

    output_folder = '../../misc/reports/{}'.format(
        datetime.datetime.now().strftime("%Y_%m%d_%H%M%S"))

    predictions_dict['m1'] = single_fitting_cycle(**copy.deepcopy(config['fitting'])) 

    m2_params = copy.deepcopy(config['fitting'])
    m2_params['split']['val_period'] = 0
    predictions_dict['m2'] = single_fitting_cycle(**m2_params)

    predictions_dict['fitting_date'] = datetime.datetime.now().strftime("%Y-%m-%d")

    predictions_dict['m1']['plots']['sensitivity'], _, _ = calculate_sensitivity_and_plot(
        predictions_dict, config, which_fit='m1')
    predictions_dict['m2']['plots']['sensitivity'], _, _ = calculate_sensitivity_and_plot(
        predictions_dict, config, which_fit='m2')

    predictions_dict['m2']['forecasts'] = {}

    predictions_dict['m2']['forecasts']['best'] = get_forecast(
        predictions_dict, train_fit='m2',
        model=config['fitting']['model'],
        days=config['forecast']['forecast_days']
    )

    predictions_dict['m2']['plots']['forecast_best'] = plot_forecast(
        predictions_dict,
        config['fitting']['data']['dataloading_params']['location_description'], 
        error_bars=True
    )

    predictions_dict['m1']['trials_processed'] = forecast_all_trials(
        predictions_dict, train_fit='m1',
        model=config['fitting']['model'],
        forecast_days=config['forecast']['forecast_days']
    )

    predictions_dict['m2']['trials_processed'] = forecast_all_trials(
        predictions_dict, train_fit='m2',
        model=config['fitting']['model'],
        forecast_days=config['forecast']['forecast_days']
    )

    kforecasts = plot_top_k_trials(
        predictions_dict, train_fit='m2',
        k=config['forecast']['num_trials_to_plot'],
        which_compartments=config['forecast']['plot_topk_trials_for_columns']
    )

    predictions_dict['m2']['plots']['forecasts_topk'] = {}
    for column in config['forecast']['plot_topk_trials_for_columns']:
        predictions_dict['m2']['plots']['forecasts_topk'][column.name] = kforecasts[column]

    uncertainty_args = {'predictions_dict': predictions_dict,
                        **config['uncertainty']['uncertainty_params']}

    uncertainty = config['uncertainty']['method'](**uncertainty_args)
    uncertainty_forecasts = uncertainty.get_forecasts()
    for key in uncertainty_forecasts.keys():
        predictions_dict['m2']['forecasts'][key] = uncertainty_forecasts[key]['df_prediction']
        
    predictions_dict['m2']['forecasts']['ensemble_mean'] = uncertainty.ensemble_mean_forecast

    predictions_dict['m2']['beta'] = uncertainty.beta
    predictions_dict['m2']['beta_loss'] = uncertainty.beta_loss
    predictions_dict['m2']['deciles'] = uncertainty_forecasts

    predictions_dict['m2']['plots']['forecast_best_50'] = plot_forecast(
        predictions_dict,
        config['fitting']['data']['dataloading_params']['location_description'],
        fits_to_plot=['best', 50], error_bars=False
    )

    predictions_dict['m2']['plots']['forecast_best_80'] = plot_forecast(
        predictions_dict,
        config['fitting']['data']['dataloading_params']['location_description'],
        fits_to_plot=['best', 80], error_bars=False
    )

    predictions_dict['m2']['plots']['forecast_ensemble_mean_50'] = plot_forecast(
        predictions_dict,
        config['fitting']['data']['dataloading_params']['location_description'],
        fits_to_plot=['ensemble_mean', 50], error_bars=False
    )

    ptiles_plots = plot_ptiles(predictions_dict, 
                               which_compartments=config['forecast']['plot_ptiles_for_columns'])
    predictions_dict['m2']['plots']['forecasts_ptiles'] = {}
    for column in config['forecast']['plot_ptiles_for_columns']:
        predictions_dict['m2']['plots']['forecasts_ptiles'][column.name] = ptiles_plots[column]

    run = wandb.init(project="covid-modelling", reinit=True, 
                     config=wandb_config, name=run_name)
    log_wandb(predictions_dict)
    run.finish()
    return predictions_dict


def perform_batch_runs(base_config_filename='us.yaml', username='sansiddh', output_folder=None):
    # Getting list of all states
    from data.dataloader import JHULoader
    obj = JHULoader()
    df = obj._load_from_daily_reports()
    what_to_vary = pd.unique(df['Province_State']).tolist()

    # Specifying the folder where checkpoints will be saved
    predictions_dict = {}
    if output_folder is None:
        output_folder = '/scratch/users/{}/covid-modelling/{}'.format(
            username, datetime.datetime.now().strftime("%Y_%m%d_%H%M%S"))
    os.makedirs(output_folder, exist_ok=True)

    for i, state in enumerate(what_to_vary):
        # Update config with new state name
        wandb_config = read_config(base_config_filename, preprocess=False)
        wandb_config['fitting']['data']['dataloading_params']['region'] = state
        config = process_config(wandb_config)
        wandb_config = make_date_key_str(wandb_config)
        print(f'Starting fitting, forecasting cycle for {state}, #{i+1}/{len(what_to_vary)}')
        try:
            predictions_dict[state] = run_single_config_end_to_end(
                config, wandb_config, f'{state}-2')
        except Exception as e:
            print(e)
        print(f'Finished cycle for {state}, #{i+1}/{len(what_to_vary)}')
        with open(f'{output_folder}/predictions_dict.pkl', 'wb') as f:
            pickle.dump(predictions_dict, f)
        print(f'Saved predictions_dict. {i+1}/{len(what_to_vary)} done.')
        plt.close('all')

if __name__ == '__main__':
    perform_batch_runs('us.yaml')
        
