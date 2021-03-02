import argparse
import json
import logging
import os
import pickle
import sys
from functools import partial
import itertools
import datetime

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

sys.path.append('../../')

from main.seir.common import *
from utils.fitting.util import update_dict, chunked
from utils.generic.config import read_config, process_config_seir

default_loss_methods = ['mape', 'rmse', 'rmse_log']


def create_output(predictions_dict, output_folder, tag):
    """Custom output generation function"""
    directory = f'{output_folder}/{tag}'
    if not os.path.exists(directory):
        os.mkdir(directory)
    d = {}
    for outer in ['m1', 'm2']:
        for inner in ['variable_param_ranges', 'best_params', 'beta_loss']:
            if inner in predictions_dict[outer]:
                with open(f'{directory}/{outer}_{inner}.json', 'w') as f:
                    json.dump(predictions_dict[outer][inner], f, indent=4)
        for inner in ['df_prediction', 'df_district', 'df_train', 'df_val', 'df_loss', 'df_district_unsmoothed']:
            if inner in predictions_dict[outer] and predictions_dict[outer][inner] is not None:
                predictions_dict[outer][inner].to_csv(f'{directory}/{outer}_{inner}.csv')
        for inner in ['trials', 'run_params', 'optimiser', 'plots', 'smoothing_description', 'default_params', ]:
            with open(f'{directory}/{outer}_{inner}.pkl', 'wb') as f:
                pickle.dump(predictions_dict[outer][inner], f)
        if 'ensemble_mean' in predictions_dict[outer]['forecasts']:
            predictions_dict[outer]['forecasts']['ensemble_mean'].to_csv(
                f'{directory}/{outer}_ensemble_mean_forecast.csv')
        predictions_dict[outer]['trials_processed']['predictions'][0].to_csv(
            f'{directory}/{outer}_trials_processed_predictions.csv')
        np.save(f'{directory}/{outer}_trials_processed_params.npy',
                predictions_dict[outer]['trials_processed']['params'])
        np.save(f'{directory}/{outer}_trials_processed_losses.npy',
                predictions_dict[outer]['trials_processed']['losses'])
        d[f'{outer}_data_last_date'] = predictions_dict[outer]['data_last_date']
    d['fitting_date'] = predictions_dict['fitting_date']
    np.save(f'{directory}/m2_beta.npy', predictions_dict['m2']['beta'])
    with open(f'{directory}/other.json', 'w') as f:
        json.dump(d, f, indent=4)


def get_experiment(which, regions, loss_methods=None, regionwise=False):
    """Get experiment configuration"""
    logging.info('Getting experiment choices')

    # Set values
    loss_methods = default_loss_methods if loss_methods is None else loss_methods
    configs = {}

    # Select experiment
    if which == 'train_lengths':
        for region in regions:
            for tl in itertools.product(np.arange(6, 45, 3), np.arange(2, 7, 1)):
                config = {
                    'fitting': {
                        'data': {'dataloading_params': region},
                        'split': {'train_period': tl[0], 'val_period': tl[1]}
                    }
                }
                configs[region['label'] + f'-{tl[0]}-{tl[1]}'] = config

    elif which == 'num_trials':
        for region in regions:
            for tl in itertools.product([21, 30], [3]):
                for i, num in enumerate([5000] * 5):
                    config = {
                        'fitting': {
                            'data': {'dataloading_params': region},
                            'fitting_method_params': {'num_evals': num},
                            'split': {'train_period': tl[0], 'val_period': tl[1]}
                        }
                    }
                    configs[region['label'] + f'-{tl[0]}-{tl[1]}' + f'-{num}-{i}'] = config

    elif which == 'num_trials_ensemble':
        for region in regions:
            for tl in itertools.product([21, 30], [3]):
                for num in np.arange(500, 5500, 500):
                    config = {
                        'fitting': {
                            'data': {'dataloading_params': region},
                            'fitting_method_params': {'num_evals': num},
                            'split': {'train_period': tl[0], 'val_period': tl[1]}
                        }
                    }
                    configs[region['label'] + f'-{tl[0]}-{tl[1]}' + f'-{num}'] = config

    elif which == 'optimiser':
        for region in regions:
            for tl in itertools.product([21, 30], [3]):
                for method, num in [('tpe', 3000), ('rand', 10000)]:
                    config = {
                        'fitting': {
                            'data': {'dataloading_params': region},
                            'fitting_method_params': {'algo': method, 'num_evals': num},
                            'split': {'train_period': tl[0], 'val_period': tl[1]}
                        }
                    }
                    configs[region['label'] + f'-{tl[0]}-{tl[1]}-{method}'] = config

    elif which == 'loss_method':
        for region in regions:
            for tl in itertools.product([30], [3]):
                for l1, l2 in itertools.product(loss_methods, loss_methods):
                    config = {
                        'fitting': {
                            'data': {'dataloading_params': region},
                            'loss': {'loss_method': l1}
                        },
                        'uncertainty': {
                            'uncertainty_params': {
                                'loss': {'loss_method': l2}
                            }
                        }
                    }
                    configs[region['label'] + f'-{tl[0]}-{tl[1]}-{l1}-{l2}'] = config

    elif which == 'windows':
        today = datetime.datetime.now().date()
        for key, region in regions.items():
            start = region['start_date']
            while start < today - datetime.timedelta(region['data_length']):
                config = {
                    'fitting': {
                        'data': {'dataloading_params': region},
                        'split': {
                            'start_date': start,
                            'end_date': None
                        }
                    },
                    'uncertainty': {
                        'date_of_sorting_trials': start+datetime.timedelta(region['data_length']-1)
                    }
                }
                start_str = start.strftime('%Y-%m-%d')
                start = start + datetime.timedelta(1)
                configs[region['label'] + '_' + start_str] = config

    if regionwise:
        configs_regionwise = {}
        for region in regions:
            configs_regionwise[region['label']] = {k: v for k, v in configs.items() if region['label'] in k}
        return configs_regionwise

    return configs


def run(config):
    """Run single experiment for given config"""
    predictions_dict = {}
    fitting(predictions_dict, config)
    predictions_dict['m1']['forecasts'] = {}
    predictions_dict['m2']['forecasts'] = {}
    uncertainty = fit_beta(predictions_dict, config)
    process_ensemble(predictions_dict, uncertainty)
    plot_ensemble_forecasts(predictions_dict, config)
    return predictions_dict


def run_parallel(run_name, params, base_config_filename):
    """Read config and run corresponding experiment"""
    config = read_config(base_config_filename, preprocess=False)
    config = update_dict(config, params)
    config = process_config_seir(config)
    try:
        logging.info(f'Start run: {run_name}')
        x = run(config)
        plt.close('all')
    except Exception as e:
        x = e
        logging.error(e)
    return x


def perform_batch_runs(base_config_filename='param_choices.yaml', driver_config_filename='list_of_exp.yaml',
                       experiment_name='train_lengths', output_folder=None):
    """Run all experiments"""
    # Specifying the folder where checkpoints will be saved
    timestamp = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
    if output_folder is None:
        output_folder = f'../../outputs/param_choices/{timestamp}'
    os.makedirs(output_folder, exist_ok=True)
    n_jobs = multiprocessing.cpu_count()

    # Get experiment choices
    regions = read_config(driver_config_filename, preprocess=False, config_dir='other')
    what_to_vary = get_experiment(experiment_name, regions)

    # Run experiments
    for i, chunk in enumerate(chunked(what_to_vary.items(), n_jobs)):
        print(f'Group {i}')
        partial_run_parallel = partial(run_parallel, base_config_filename=base_config_filename)
        logging.info('Start batch runs')
        predictions_arr = Parallel(n_jobs=n_jobs)(
            delayed(partial_run_parallel)(key, config) for key, config in tqdm(chunk.items()))

        for j, key in tqdm(enumerate(chunk.keys())):
            if isinstance(predictions_arr[j], dict):
                create_output(predictions_arr[j], output_folder, key)
                # with open(f'{output_folder}/{key}_predictions_dict.pkl', 'wb') as f:
                #     pickle.dump(predictions_arr[j], f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description="SEIR Batch Running Script")
    parser.add_argument("--base_config", type=str, required=True, help="base config to use while running the script")
    parser.add_argument("--driver_config", type=str, required=True, help="driver config used for multiple experiments")
    parser.add_argument("--experiment", type=str, required=True, help="experiment name")
    parsed_args = parser.parse_args()
    perform_batch_runs(parsed_args.base_config, parsed_args.driver_config, parsed_args.experiment)
