import argparse
import logging
import os
import pickle
import sys
from functools import partial
import itertools
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

sys.path.append('../../')

from main.seir.common import *
from utils.fitting.util import update_dict, chunked
from utils.generic.config import read_config, process_config

default_loss_methods = ['mape', 'rmse', 'rmse_log']


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
                for i, num in enumerate([5000]*5):
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
        today = datetime.today()
        for region in regions:
            start = datetime.strptime(region['start_date'], '%Y-%m-%d')
            while start < today-timedelta(region['data_length']):
                config = {
                    'fitting': {
                        'data': {'dataloading_params': region},
                        'split': {
                            'start_date': start,
                            'end_date': None
                        }
                    }
                }
                start = start + timedelta(1)
                start_str = start.strftime('%Y-%m-%d')
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
    config = process_config(config)
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
                with open(f'{output_folder}/{key}_predictions_dict.pkl', 'wb') as f:
                    pickle.dump(predictions_arr[j], f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description="SEIR Batch Running Script")
    parser.add_argument("--base_config", type=str, required=True, help="base config to use while running the script")
    parser.add_argument("--driver_config", type=str, required=False, help="driver config used for multiple experiments")
    parser.add_argument("--experiment", type=str, required=True, help="experiment name")
    parsed_args = parser.parse_args()
    perform_batch_runs(parsed_args.base_config, parsed_args.driver_config, parsed_args.experiment)
