import argparse
import copy
import datetime
import itertools
import logging
import os
import pickle
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append('../../')

from main.seir.fitting import single_fitting_cycle
from main.seir.forecast import get_forecast
from main.seir.sensitivity import calculate_sensitivity_and_plot
from viz import plot_forecast, plot_top_k_trials, plot_ptiles
from viz.uncertainty import plot_beta_loss
from utils.fitting.util import update_dict
from utils.generic.config import read_config, process_config

regions = [
    {'label': 'Mumbai', 'state': 'Maharashtra', 'district': 'Mumbai', 'smooth_jump': True},
    {'label': 'Delhi', 'state': 'Delhi', 'district': None},
    {'label': 'Kerala', 'state': 'Kerala', 'district': None},
    {'label': 'Bengaluru', 'state': 'Karnataka', 'district': 'Bengaluru Urban'},
    {'label': 'Pune', 'state': 'Maharashtra', 'district': 'Pune'},
    {'label': 'Chennai', 'state': 'Tamilnadu', 'district': 'Chennai'}
]


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


def process_beta_fitting(predictions_dict, uncertainty):
    predictions_dict['m2']['plots']['beta_loss'], _ = plot_beta_loss(
        uncertainty.dict_of_trials)

    predictions_dict['m2']['forecasts']['ensemble_mean'] = uncertainty.ensemble_mean_forecast

    predictions_dict['m2']['beta'] = uncertainty.beta
    predictions_dict['m2']['beta_loss'] = uncertainty.beta_loss


def process_uncertainty_fitting(predictions_dict, uncertainty):
    uncertainty_forecasts = uncertainty.get_forecasts()
    for key in uncertainty_forecasts.keys():
        predictions_dict['m2']['forecasts'][key] = uncertainty_forecasts[key]['df_prediction']
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


def plot_ensemble_forecasts(predictions_dict, config):
    predictions_dict['m2']['plots']['forecast_ensemble_mean'] = plot_forecast(
        predictions_dict,
        config['fitting']['data']['dataloading_params']['location_description'],
        which_compartments=config['fitting']['loss']['loss_compartments'],
        fits_to_plot=['ensemble_mean'], error_bars=False
    )


def plot_forecasts_ptiles(predictions_dict, config):
    ptiles_plots = plot_ptiles(predictions_dict,
                               which_compartments=config['forecast']['plot_ptiles_for_columns'])
    predictions_dict['m2']['plots']['forecasts_ptiles'] = {}
    for column in config['forecast']['plot_ptiles_for_columns']:
        predictions_dict['m2']['plots']['forecasts_ptiles'][column.name] = ptiles_plots[column]


def get_experiment(which, regionwise=True):
    logging.info('Getting experiment choices')
    configs = {}
    if which == 'train_lengths':
        for region in regions:
            for tl in itertools.product(np.arange(6, 45, 3), np.arange(2, 6, 1)):
                config = {'fitting':
                              {'data':
                                   {'dataloading_params': region},
                               'split':
                                   {'train_period': tl[0], 'val_period': tl[1]}
                               }
                          }
                configs[region['label'] + f'-{tl[0]}-{tl[1]}'] = config

    if regionwise:
        configs_regionwise = {}
        for region in regions:
            configs_regionwise[region['label']] = {k: v for k, v in configs.items() if region['label'] in k}
        return configs_regionwise

    return configs


def run(config):
    predictions_dict = {}
    fitting(predictions_dict, config)
    predictions_dict['m1']['forecasts'] = {}
    predictions_dict['m2']['forecasts'] = {}
    uncertainty = fit_beta(predictions_dict, config)
    process_beta_fitting(predictions_dict, uncertainty)
    plot_ensemble_forecasts(predictions_dict, config)
    return predictions_dict


def run_parallel(run_name, params, base_config_filename):
    config = read_config(base_config_filename, preprocess=False)
    config = update_dict(config, params)
    print(config['fitting'])
    config = process_config(config)
    try:
        logging.info(f'Start run: {run_name}')
        x = run(config)
        plt.close('all')
    except Exception as e:
        x = e
        logging.error(e)
    return x


def perform_batch_runs(base_config_filename='param_choices.yaml', experiment_name='train_lengths', output_folder=None):
    # Specifying the folder where checkpoints will be saved
    timestamp = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
    if output_folder is None:
        output_folder = f'../../outputs/param_choices/{timestamp}'
    os.makedirs(output_folder, exist_ok=True)

    # Get experiment choices
    what_to_vary = get_experiment(experiment_name)
    for k, v in what_to_vary.items():
        partial_run_parallel = partial(run_parallel, base_config_filename=base_config_filename)
        logging.info('Start batch runs')
        predictions_arr = Parallel(n_jobs=40)(
            delayed(partial_run_parallel)(key, config) for key, config in tqdm(v.items()))

        predictions_dict = {}
        for i, (key, config) in tqdm(enumerate(v.items())):
            if type(predictions_arr[i]) == dict:
                predictions_dict[key] = predictions_arr[i]
        with open(f'{output_folder}/{k}_predictions_dict.pkl', 'wb') as f:
            pickle.dump(predictions_dict, f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description="SEIR Batch Running Script")
    parser.add_argument("--filename", type=str, required=True, help="config filename to use while running the script")
    parser.add_argument("--experiment", type=str, required=True, help="experiment name")
    parsed_args = parser.parse_args()
    perform_batch_runs(parsed_args.filename, parsed_args.experiment)
