"""
experiments.py
Run hyperparameter tuning and other experiments for variants of compartmental models.
"""

import argparse
import datetime
import itertools
import json
import logging
import os
import pickle
import sys
import copy

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

sys.path.append('../../')

from main.seir.fitting import single_fitting_cycle
from utils.fitting.util import update_dict, chunked, CustomEncoder
from utils.generic.config import read_config, process_config_seir, make_date_str, get_configs_from_driver


def create_output(predictions_dict, output_folder, tag):
    """Custom output generation function"""
    directory = f'{output_folder}/{tag}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    d = {}
    for key in ['variable_param_ranges', 'best_params', 'beta_loss']:
        if key in predictions_dict:
            with open(f'{directory}/{key}.json', 'w') as f:
                json.dump(predictions_dict[key], f, indent=4)
    for key in ['df_prediction', 'df_district', 'df_train', 'df_val', 'df_loss', 'df_district_unsmoothed']:
        if key in predictions_dict and predictions_dict[key] is not None:
            predictions_dict[key].to_csv(f'{directory}/{key}.csv')
    for key in ['trials', 'run_params', 'optimiser', 'plots', 'smoothing_description', 'default_params']:
        with open(f'{directory}/{key}.pkl', 'wb') as f:
            pickle.dump(predictions_dict[key], f)
    d[f'data_last_date'] = predictions_dict['data_last_date']
    d['fitting_date'] = predictions_dict['fitting_date']
    with open(f'{directory}/other.json', 'w') as f:
        json.dump(d, f, indent=4)
    with open(f'{directory}/config.json', 'w') as f:
        json.dump(make_date_str(predictions_dict['config']), f, indent=4, cls=CustomEncoder)


def get_experiment(driver_config_filename):
    """Get experiment configuration"""
    logging.info('Getting experiment choices')
    configs = get_configs_from_driver(driver_config_filename)
    return configs


def run(config):
    """Run single experiment for given config"""
    predictions_dict = single_fitting_cycle(
        **copy.deepcopy(config['fitting']))
    predictions_dict['fitting_date'] = datetime.datetime.now().strftime(
        "%Y-%m-%d")
    return predictions_dict


def run_parallel(key, params):
    """Read config and run corresponding experiment"""
    config = read_config(f'{key}.yaml', preprocess=False)
    config = update_dict(config, params)
    config = process_config_seir(config)
    try:
        logging.info(f'Start run: {key}')
        x = run(config)
        x['config'] = config
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
        output_folder = f'../../outputs/seir/{timestamp}'
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
                create_output(predictions_arr[j], output_folder, f'{key}/{n_jobs*i+j}')
                # with open(f'{output_folder}/{key}_predictions_dict.pkl', 'wb') as f:
                #     pickle.dump(predictions_arr[j], f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description="SEIR Batch Running Script")
    parser.add_argument("-d", "--driver_config", type=str, required=True,
                        help="driver config used for multiple experiments")
    parsed_args = parser.parse_args()
    perform_batch_runs(parsed_args.driver_config)
