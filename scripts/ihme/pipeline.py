"""
pipeline.py
Run a single fitting cycle using the IHME model.
Loads data, optimizes initial parameters, fits the IHME model to the data and evaluates on the train, val and test sets.
Run as: python3 -W ignore pipeline.py -c default.yaml
"""
import argparse
import copy
import datetime
import json
import os
import pickle
import sys
import warnings
import pandas as pd
import numpy as np
from tabulate import tabulate

sys.path.append('../../')

from main.ihme.fitting import single_fitting_cycle
from utils.fitting.util import CustomEncoder
from utils.generic.config import read_config, make_date_str

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', module='pandas', category=RuntimeWarning)
warnings.filterwarnings('ignore', module='curvefit', category=RuntimeWarning)
warnings.filterwarnings('ignore', module='numpy', category=RuntimeWarning)


def create_output(predictions_dict, output_folder, config):
    """Custom output generation function"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    d = {}
    # Numpy
    for key in ['best_init', 'best_params', 'draws']:
        np.save(f'{output_folder}/{key}.npy', predictions_dict[key])
    # Pickle
    for key in ['trials', 'run_params', 'plots', 'smoothing_description']:
        with open(f'{output_folder}/{key}.pkl', 'wb') as f:
            pickle.dump(predictions_dict[key], f)
    # Dataframes
    for key in ['df_prediction', 'df_district', 'df_train', 'df_val', 'df_loss', 'df_loss_pointwise',
                'df_district_unsmoothed', 'df_train_nora_notrans', 'df_val_nora_notrans', 'df_test_nora_notrans']:
        if predictions_dict[key] is not None:
            predictions_dict[key].to_csv(f'{output_folder}/{key}.csv')
    # JSON
    d['data_last_date'] = predictions_dict['data_last_date']
    with open(f'{output_folder}/other.json', 'w') as f:
        json.dump(d, f, indent=4)
    with open(f'{output_folder}/config.json', 'w') as f:
        json.dump(make_date_str(config), f, indent=4, cls=CustomEncoder)


def run_pipeline(config_filename):
    """Run pipeline for a given config

    Args:
        config_filename (str): config file name

    """
    config = read_config(config_filename, preprocess=True, config_dir='ihme')
    timestamp = datetime.datetime.now()
    output_folder = '../../misc/ihme/{}'.format(timestamp.strftime("%Y_%m%d_%H%M%S"))
    predictions_dict = single_fitting_cycle(**copy.deepcopy(config['fitting']))
    print('loss\n', tabulate(predictions_dict['df_loss'].tail().round(2).T, headers='keys', tablefmt='psql'))
    create_output(predictions_dict, output_folder, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config file name", required=True)
    args = parser.parse_args()

    run_pipeline(args.config)
