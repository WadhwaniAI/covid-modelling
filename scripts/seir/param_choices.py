import argparse
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

from main.seir.common import *
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
