import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import argparse
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm

import sys
sys.path.append('../../')

from main.seir.common import *
from utils.generic.config import read_config, process_config, make_date_key_str
from utils.generic.logging import log_wandb
from viz import plot_forecast

import wandb


def plot_forecasts_of_best_candidates(predictions_dict, config):
    predictions_dict['m2']['plots']['forecast_best_50'] = plot_forecast(
        predictions_dict,
        config['fitting']['data']['dataloading_params']['location_description'],
        which_compartments=config['fitting']['loss']['loss_compartments'],
        fits_to_plot=['best', 50], error_bars=False
    )

    predictions_dict['m2']['plots']['forecast_best_80'] = plot_forecast(
        predictions_dict,
        config['fitting']['data']['dataloading_params']['location_description'],
        which_compartments=config['fitting']['loss']['loss_compartments'],
        fits_to_plot=['best', 80], error_bars=False
    )

    predictions_dict['m2']['plots']['forecast_ensemble_mean_50'] = plot_forecast(
        predictions_dict,
        config['fitting']['data']['dataloading_params']['location_description'],
        which_compartments=config['fitting']['loss']['loss_compartments'],
        fits_to_plot=['ensemble_mean', 50], error_bars=False
    )


def run_single_config_end_to_end(config, wandb_config, run_name, log_wandb_flag=False):
    predictions_dict = {}

    fitting(predictions_dict, config)
    forecast_best(predictions_dict, config)
    uncertainty = fit_beta(predictions_dict, config)
    process_uncertainty_fitting(predictions_dict, uncertainty)

    plot_forecasts_top_k_trials(predictions_dict, config)
    plot_forecasts_of_best_candidates(predictions_dict, config)
    plot_forecasts_ptiles(predictions_dict, config)

    if log_wandb_flag:
        run = wandb.init(project="covid-modelling", reinit=True, 
                        config=wandb_config, name=run_name)
        log_wandb(predictions_dict)
        run.finish()
    return predictions_dict


def single_run_fn_for_parallel(state, base_config_filename):
    wandb_config = read_config(base_config_filename, preprocess=False)
    wandb_config['fitting']['data']['dataloading_params']['region'] = state
    config = process_config(wandb_config)
    wandb_config = make_date_key_str(wandb_config)
    try:
        x = run_single_config_end_to_end(
            config, wandb_config, f'{state}', log_wandb_flag=False)
        plt.close('all')
    except Exception as e:
        x = e
    return x
    

def perform_batch_runs(base_config_filename='us.yaml', username='sansiddh', output_folder=None):
    # Specifying the folder where checkpoints will be saved
    if output_folder is None:
        output_folder = '/scratche/users/{}/covid-modelling/{}'.format(
            username, datetime.datetime.now().strftime("%Y_%m%d_%H%M%S"))
    os.makedirs(output_folder, exist_ok=True)

    # Getting list of all states
    print('Getting list of all states')
    from data.dataloader import JHULoader
    obj = JHULoader()
    df = obj._load_from_daily_reports()
    what_to_vary = pd.unique(df['Province_State']).tolist()
    partial_single_run_fn_for_parallel = partial(
        single_run_fn_for_parallel, base_config_filename=base_config_filename)

    predictions_arr = Parallel(n_jobs=40)(delayed(partial_single_run_fn_for_parallel)(state)
                                          for state in tqdm(what_to_vary))
    predictions_dict = {}
    for i, state in tqdm(enumerate(what_to_vary)):
        if type(predictions_arr[i]) == dict:
            predictions_dict[state] = predictions_arr[i]
    with open(f'{output_folder}/predictions_dict.pkl', 'wb') as f:
        pickle.dump(predictions_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SEIR Batch Running Script")
    parser.add_argument(
        "--filename", type=str, required=True, help="config filename to use while running the script"
    )
    parsed_args = parser.parse_args()
    perform_batch_runs(parsed_args.filename)
        
