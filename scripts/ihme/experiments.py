"""
experiments.py
Run multiple experiments with multiple configurations.
Create a new config under configs/ihme for each region.
Create a driver config specifying the experiments under configs/other.
Run as: python3 -W ignore experiments.py -b region.yaml -d driver.yaml
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

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

sys.path.append('../../')

from main.ihme.fitting import single_fitting_cycle
from utils.fitting.util import update_dict, chunked, CustomEncoder
from utils.generic.config import read_config, process_config_ihme, make_date_str, generate_combinations, generate_config

default_loss_methods = ['mape', 'rmse', 'rmse_log']


def create_output(predictions_dict, output_folder, tag):
    """Custom output generation function"""
    directory = f'{output_folder}/{tag}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    d = {}
    # Numpy
    for key in ['best_init', 'best_params', 'draws']:
        np.save(f'{directory}/{key}.npy', predictions_dict[key])
    # Pickle
    for key in ['trials', 'run_params', 'plots', 'smoothing_description']:
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
    with open(f'{directory}/config.json', 'w') as f:
        json.dump(make_date_str(predictions_dict['config']), f, indent=4, cls=CustomEncoder)


def get_experiments(driver_config_filename, base_config_filename):
    """Get experiment configuration"""
    logging.info('Getting experiment choices')
    experiments = read_config(driver_config_filename, preprocess=False, config_dir='other')
    label = base_config_filename.split('.')[0]
    experiments = generate_config(experiments)
    experiments = generate_combinations(experiments)
    configs = {f'{label}/{i}': exp for i, exp in enumerate(experiments)}
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
        x['config'] = config
        plt.close('all')
    except Exception as e:
        x = e
        logging.error(e)
    return x


def perform_batch_runs(base_config_filename='default.yaml', driver_config_filename='list_of_exp.yaml',
                       output_folder=None):
    """Run all experiments"""
    # Specifying the folder where checkpoints will be saved
    timestamp = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
    if output_folder is None:
        output_folder = f'../../outputs/ihme/{timestamp}'
    os.makedirs(output_folder, exist_ok=True)
    n_jobs = multiprocessing.cpu_count()

    # Get experiment choices
    what_to_vary = get_experiments(driver_config_filename, base_config_filename)

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
                # with open(f'{output_folder}/{key}/predictions_dict.pkl', 'wb') as f:
                #     pickle.dump(predictions_arr[j], f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description="IHME Batch Running Script")
    parser.add_argument("-b", "--base_config", type=str, required=True,
                        help="base config to use while running the script")
    parser.add_argument("-d", "--driver_config", type=str, required=False, default='list_of_exp.yaml',
                        help="driver config used for multiple experiments")
    parsed_args = parser.parse_args()
    perform_batch_runs(parsed_args.base_config, parsed_args.driver_config)
