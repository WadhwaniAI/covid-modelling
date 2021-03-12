"""
experiments.py
Run experiments with multiple configurations.
Create a driver config specifying the experiments under configs/other.
Run as: python3 -W ignore experiments.py -b region.yaml -d driver.yaml
"""
import argparse
import copy
import datetime
import itertools
import json
import logging
import multiprocessing
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append('../../')

from main.ihme.fitting import single_fitting_cycle
from utils.fitting.util import update_dict, chunked, CustomEncoder
from utils.generic.config import read_config, process_config_ihme, make_date_str, generate_configs_from_driver

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
    with open(f'{directory}/config.yaml', 'w') as f:
        yaml.dump(make_date_str(predictions_dict['config']), f)


def get_experiment(driver_config_filename):
    """Get experiment configuration"""
    logging.info('Getting experiment choices')
    configs = generate_configs_from_driver(driver_config_filename)
    return configs


def run(config):
    """Run single experiment for given config"""
    predictions_dict = single_fitting_cycle(**copy.deepcopy(config['fitting']))
    predictions_dict['fitting_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
    return predictions_dict


def run_parallel(key, params):
    """Read config and run corresponding experiment"""
    config = read_config(f'{key}.yaml', preprocess=False, config_dir='ihme')
    config = update_dict(config, params)
    config_copy = copy.deepcopy(config)
    config = process_config_ihme(config)
    try:
        logging.info(f'Start run: {key}')
        x = run(config)
        x['config'] = config_copy
        plt.close('all')
    except Exception as e:
        x = e
        logging.error(e)
    return x


def perform_batch_runs(driver_config_filename='list_of_exp.yaml', output_folder=None):
    """Run all experiments"""
    # Specifying the folder where checkpoints will be saved
    timestamp = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
    if output_folder is None:
        output_folder = f'../../outputs/ihme/{timestamp}'
    os.makedirs(output_folder, exist_ok=True)
    n_jobs = multiprocessing.cpu_count()

    # Get generator of partial configs corresponding to experiments
    what_to_vary = get_experiment(driver_config_filename)

    # Run experiments
    logging.info('Start batch runs')
    for i, chunk in enumerate(chunked(what_to_vary, n_jobs)):
        chunk1, chunk2 = itertools.tee(chunk, 2)
        print(f'Group {i}')
        predictions_arr = Parallel(n_jobs=n_jobs)(
            delayed(run_parallel)(key, config) for key, config in tqdm(chunk1))

        # Save results
        for j, (key, _) in tqdm(enumerate(chunk2)):
            if isinstance(predictions_arr[j], dict):
                create_output(predictions_arr[j], output_folder, f'{key}/{n_jobs * i + j}')
                # with open(f'{output_folder}/{key}_predictions_dict.pkl', 'wb') as f:
                #     pickle.dump(predictions_arr[j], f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description="IHME Batch Running Script")
    parser.add_argument("-d", "--driver_config", type=str, required=True,
                        help="driver config used for multiple experiments")
    parsed_args = parser.parse_args()
    perform_batch_runs(parsed_args.driver_config)
