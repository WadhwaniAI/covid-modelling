"""
experiments.py
Run hyperparameter tuning and other experiments for variants of compartmental models using the ABMA approach.
"""
import argparse
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

from scripts.seir.common import *
from utils.fitting.util import CustomEncoder, chunked, update_dict
from utils.generic.config import (
    generate_configs_from_driver,
    make_date_str,
    process_config_seir,
    read_config,
)


def create_output(predictions_dict, output_folder, tag):
    """Custom output generation function"""
    directory = f'{output_folder}/{tag}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    d = {}
    for inner in ['variable_param_ranges', 'best_params', 'beta_loss']:
        if inner in predictions_dict:
            with open(f'{directory}/{inner}.json', 'w') as f:
                json.dump(predictions_dict[inner], f, indent=4)
    for inner in ['df_prediction', 'df_district', 'df_train', 'df_val', 'df_loss', 'df_district_unsmoothed']:
        if inner in predictions_dict and predictions_dict[inner] is not None:
            predictions_dict[inner].to_csv(f'{directory}/{inner}.csv')
    for inner in ['trials', 'run_params', 'plots', 'smoothing_description', 'default_params']:
        with open(f'{directory}/{inner}.pkl', 'wb') as f:
            pickle.dump(predictions_dict[inner], f)
    if 'ensemble_mean' in predictions_dict['forecasts']:
        predictions_dict['forecasts']['ensemble_mean'].to_csv(
            f'{directory}/ensemble_mean_forecast.csv')
    predictions_dict['trials']['predictions'][0].to_csv(
        f'{directory}/trials_predictions.csv')
    np.save(f'{directory}/trials_params.npy',
            predictions_dict['trials']['params'])
    np.save(f'{directory}/trials_losses.npy',
            predictions_dict['trials']['losses'])
    d[f'data_last_date'] = predictions_dict['data_last_date']
    d['fitting_date'] = predictions_dict['fitting_date']
    np.save(f'{directory}/beta.npy', predictions_dict['beta'])
    with open(f'{directory}/other.json', 'w') as f:
        json.dump(d, f, indent=4)
    with open(f'{directory}/config.json', 'w') as f:
        json.dump(make_date_str(
            predictions_dict['config']), f, indent=4, cls=CustomEncoder)
    with open(f'{directory}/config.yaml', 'w') as f:
        yaml.dump(make_date_str(predictions_dict['config']), f)


def get_experiment(driver_config_filename):
    """Get experiment configuration"""
    logging.info('Getting experiment choices')
    configs = generate_configs_from_driver(driver_config_filename)
    return configs


def run(config):
    """Run single experiment for given config"""
    predictions_dict = fitting(config)
    predictions_dict['forecasts'] = {}
    predictions_dict['forecasts'] = {}
    uncertainty = fit_beta(predictions_dict, config)
    process_ensemble(predictions_dict, uncertainty)
    plot_ensemble_forecasts(predictions_dict, config)
    return predictions_dict


def run_parallel(key, params):
    """Read config and run corresponding experiment"""
    config = read_config(f'{key}.yaml', preprocess=False)
    config = update_dict(config, params)
    config_copy = copy.deepcopy(config)
    config = process_config_seir(config)
    # try:
    logging.info(f'Start run: {key}')
    x = run(config)
    x['config'] = config_copy
    plt.close('all')
    return x


def perform_batch_runs(driver_config_filename='list_of_exp.yaml', output_folder=None, n_jobs=None, 
                       tag='other'):
    """Run all experiments"""
    # Specifying the folder where checkpoints will be saved
    timestamp = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
    if output_folder is None:
        output_folder = '../../outputs/seir/'
    output_folder = os.path.join(output_folder, tag+'_'+timestamp)
    os.makedirs(output_folder, exist_ok=True)
    if n_jobs is None:
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
                create_output(predictions_arr[j],
                              output_folder, f'{key}/{n_jobs * i + j}')
                # with open(f'{output_folder}/{key}_predictions_dict.pkl', 'wb') as f:
                #     pickle.dump(predictions_arr[j], f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description="SEIR ABMA Batch Running Script")
    parser.add_argument("-d", "--driver_config", type=str, required=True,
                        help="driver config used for multiple experiments")
    parser.add_argument("-o", "--output_folder", type=str, required=False, default=None,
                        help="Where to save the outputs")
    parser.add_argument("--n_jobs", type=int, help="number of threads")
    parser.add_argument("--tag", type=str, help="Experiments tag name", default="other")
    parsed_args = parser.parse_args()
    perform_batch_runs(parsed_args.driver_config,
                       parsed_args.output_folder, parsed_args.n_jobs,
                       parsed_args.tag)
