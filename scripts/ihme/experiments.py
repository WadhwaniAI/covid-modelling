"""
experiments.py
"""
import argparse
import copy
import datetime
import json
import logging
import os
import pickle
import sys
from functools import partial
import itertools

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

sys.path.append('../../')

from main.ihme.fitting import single_fitting_cycle
from utils.fitting.util import update_dict, chunked
from utils.generic.config import read_config, process_config_ihme

default_loss_methods = ['mape', 'rmse', 'rmse_log']


def create_output(predictions_dict, output_folder, tag):
    """Custom output generation function"""
    directory = f'{output_folder}/{tag}'
    if not os.path.exists(directory):
        os.mkdir(directory)
    d = {}
    # Numpy
    for key in ['best_init', 'best_params', 'draws']:
        np.save(f'{directory}/{key}.npy', predictions_dict[key])
    # Pickle
    for key in ['trials', 'run_params', 'optimiser', 'plots', 'smoothing_description']:
        with open(f'{directory}/{key}.pkl', 'wb') as f:
            pickle.dump(predictions_dict[key], f)
    # Dataframes
    for key in ['df_prediction', 'df_district', 'df_train', 'df_val', 'df_loss', 'df_loss_pointwise',
                'df_district_unsmoothed', 'df_train_nora_notrans', 'df_val_nora_notrans', 'df_test_nora_notrans']:
        if predictions_dict[key] is not None:
            predictions_dict[key].to_csv(f'{directory}/{key}.csv')
    # JSON
    d['data_last_date'] = predictions_dict['data_last_date']
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
        # Optimize the length of the fitting and validation periods
        for region in regions:
            for tl in itertools.product(np.arange(6, 45, 3), np.arange(2, 7, 1)):
                config = {
                    'fitting': {
                        'data': {'dataloading_params': region,
                                 'data_source': region['data_source']},
                        'split': {'train_period': tl[0], 'val_period': tl[1]}
                    }
                }
                configs[region['label'] + f'/{tl[0]}-{tl[1]}'] = config

    elif which == 'num_trials':
        # Optimize the number of hyperopt trials for fitting
        for region in regions:
            for tl in itertools.product([21], [7]):
                for i, num in enumerate([500] * 5):
                    config = {
                        'fitting': {
                            'data': {'dataloading_params': region,
                                     'data_source': region['data_source']},
                            'fitting_method_params': {'num_evals': num},
                            'split': {'train_period': tl[0], 'val_period': tl[1]}
                        }
                    }
                    configs[region['label'] + f'/{tl[0]}-{tl[1]}' + f'-{num}-{i}'] = config

    elif which == 'loss_method':
        # Compare fitting loss methods
        for region in regions:
            for tl in itertools.product([30], [3]):
                for l in loss_methods:
                    config = {
                        'fitting': {
                            'data': {'dataloading_params': region,
                                     'data_source': region['data_source']},
                            'loss': {'loss_method': l}
                        }
                    }
                    configs[region['label'] + f'/{tl[0]}-{tl[1]}-{l}'] = config

    elif which == 'windows':
        # Fit to multiple windows
        today = datetime.datetime.now().date()
        for key, region in regions.items():
            start = region['start_date']
            while start < today - datetime.timedelta(region['data_length']):
                config = {
                    'fitting': {
                        'data': {'dataloading_params': region,
                                 'data_source': region['data_source']},
                        'split': {
                            'start_date': start,
                            'end_date': None
                        }
                    }
                }
                start_str = start.strftime('%Y-%m-%d')
                start = start + datetime.timedelta(1)
                configs[region['label'] + '/' + start_str] = config

    if regionwise:
        configs_regionwise = {}
        for region in regions:
            configs_regionwise[region['label']] = {k: v for k, v in configs.items() if region['label'] in k}
        return configs_regionwise

    return configs


def run(config):
    """Run single experiment for given config"""
    predictions_dict = single_fitting_cycle(**copy.deepcopy(config['fitting']))
    return predictions_dict


def run_parallel(run_name, params, base_config_filename):
    """Read config and run corresponding experiment"""
    config = read_config(base_config_filename, preprocess=False, config_dir='ihme')
    config = update_dict(config, params)
    config = process_config_ihme(config)
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
        output_folder = f'../../outputs/ihme/{timestamp}'
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
    parser.add_argument("-b", "--base_config", type=str, required=True,
                        help="base config to use while running the script")
    parser.add_argument("-d", "--driver_config", type=str, required=False, default='list_of_exp.yaml',
                        help="driver config used for multiple experiments")
    parser.add_argument("-e", "--experiment", type=str, required=True, help="experiment name")
    parsed_args = parser.parse_args()
    perform_batch_runs(parsed_args.base_config, parsed_args.driver_config, parsed_args.experiment)
