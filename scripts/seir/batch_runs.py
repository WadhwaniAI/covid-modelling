import matplotlib.pyplot as plt
import pandas as pd
import datetime
import copy
import os
import pickle
import argparse
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm

import sys
sys.path.append('../../')

from main.seir.fitting import single_fitting_cycle
from main.seir.forecast import get_forecast
from main.seir.sensitivity import calculate_sensitivity_and_plot
from utils.generic.config import read_config, process_config, make_date_key_str
from utils.generic.logging import log_wandb
from viz import plot_forecast, plot_top_k_trials, plot_ptiles
from viz.uncertainty import plot_beta_loss

import wandb


def fitting(predictions_dict, config):
    predictions_dict['m1'] = single_fitting_cycle(
        **copy.deepcopy(config['fitting']))

    m2_params = copy.deepcopy(config['fitting'])
    m2_params['split']['val_period'] = 0
    predictions_dict['m2'] = single_fitting_cycle(**m2_params)

    predictions_dict['fitting_date'] = datetime.datetime.now().strftime(
        "%Y-%m-%d")

def sensitivity(predictions_dict, config):
    predictions_dict['m1']['plots']['sensitivity'], _, _ = calculate_sensitivity_and_plot(
        predictions_dict, config, which_fit='m1')
    predictions_dict['m2']['plots']['sensitivity'], _, _ = calculate_sensitivity_and_plot(
        predictions_dict, config, which_fit='m2')

def fit_beta(predictions_dict, config):
    uncertainty_args = {'predictions_dict': predictions_dict,
                        'fitting_config': config['fitting'],
                        'forecast_config': config['forecast'],
                        **config['uncertainty']['uncertainty_params']}

    uncertainty = config['uncertainty']['method'](**uncertainty_args)
    return uncertainty

def process_uncertainty_fitting(predictions_dict, config, uncertainty):
    predictions_dict['m2']['plots']['beta_loss'], _ = plot_beta_loss(
        uncertainty.dict_of_trials)
    uncertainty_forecasts = uncertainty.get_forecasts()
    for key in uncertainty_forecasts.keys():
        predictions_dict['m2']['forecasts'][key] = uncertainty_forecasts[key]['df_prediction']

    predictions_dict['m2']['forecasts']['ensemble_mean'] = uncertainty.ensemble_mean_forecast

    predictions_dict['m2']['beta'] = uncertainty.beta
    predictions_dict['m2']['beta_loss'] = uncertainty.beta_loss
    predictions_dict['m2']['deciles'] = uncertainty_forecasts

def forecast_best(predictions_dict, config):
    predictions_dict['m1']['forecasts'] = {}
    predictions_dict['m2']['forecasts'] = {}
    for fit in ['m1', 'm2']:
        predictions_dict[fit]['forecasts']['best'] = get_forecast(
            predictions_dict, train_fit=fit,
            model=config['fitting']['model'],
            train_end_date=config['fitting']['split']['end_date'],
            forecast_days=config['forecast']['forecast_days']
        )

        predictions_dict[fit]['plots']['forecast_best'] = plot_forecast(
            predictions_dict,
            config['fitting']['data']['dataloading_params']['location_description'],
            which_fit=fit, error_bars=False,
            which_compartments=config['fitting']['loss']['loss_compartments']
        )

def plot_forecasts_top_k_trials(predictions_dict, config):
    kforecasts = plot_top_k_trials(
        predictions_dict, train_fit='m2',
        k=config['forecast']['num_trials_to_plot'],
        which_compartments=config['forecast']['plot_topk_trials_for_columns']
    )

    predictions_dict['m2']['plots']['forecasts_topk'] = {}
    for column in config['forecast']['plot_topk_trials_for_columns']:
        predictions_dict['m2']['plots']['forecasts_topk'][column.name] = kforecasts[column]

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

def plot_forecasts_ptiles(predictions_dict, config):
    ptiles_plots = plot_ptiles(predictions_dict,
                               which_compartments=config['forecast']['plot_ptiles_for_columns'])
    predictions_dict['m2']['plots']['forecasts_ptiles'] = {}
    for column in config['forecast']['plot_ptiles_for_columns']:
        predictions_dict['m2']['plots']['forecasts_ptiles'][column.name] = ptiles_plots[column]


def run_single_config_end_to_end(config, wandb_config, run_name, perform_sensitivity=False, log_wandb_flag=False):
    predictions_dict = {}

    output_folder = '../../misc/reports/{}'.format(
        datetime.datetime.now().strftime("%Y_%m%d_%H%M%S"))

    fitting(predictions_dict, config)
    if perform_sensitivity:
        sensitivity(predictions_dict, config)
    forecast_best(predictions_dict, config)
    uncertainty = fit_beta(predictions_dict, config)
    process_uncertainty_fitting(predictions_dict, config, uncertainty)

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
        output_folder = '/scratch/users/{}/covid-modelling/{}'.format(
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
        