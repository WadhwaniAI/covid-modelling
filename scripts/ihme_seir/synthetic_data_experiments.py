"""
Script to run IHME-SEIR synthetic data experiments. Executed using runner.sh to perform multiple runs.
Data generators:
    IHME I1 model (1)
    Compartmental C1 model
        SEIR Testing model (2)
        SIRD model (3)
Forecasting models:
    C2 model:
        SEIR Testing model:
            Using ground truth data (4)
            Using synthetic data from I1 (5)
            Using synthetic data from C1 (6)
        SIRD model:
            Using ground truth data (7)
            Using synthetic data from I1 (8)
            Using synthetic data from C1 (9)
Baseline models:
    C3 model: Using ground truth from s1 only
        SEIR Testing model (10)
        SIRD model (11)

Usage example (single run):
    python3 synthetic_data_experiments.py -i ../ihme/config/pune.yaml -r config/pune.yaml -f test -n 2
"""

import argparse
import sys

sys.path.append('../../')

from main.ihme_seir.experiments import run_experiments

from data.processing import get_data

from main.ihme_seir.synthetic_data_generator import read_region_config


def runner(ihme_config_path, region_config_path, output_folder, num):
    """Performs a single run of experiments for a region

    Args:
        ihme_config_path (str): path to ihme config
        region_config_path (str): path to experiment config for region
        output_folder (str): path to output folder
        num (int): number of periods to shift forward to get start date of experiments,
            where shift is specified in the config for the region

    """

    region_config = read_region_config(region_config_path)
    district = region_config['district']
    state = region_config['state']
    disable_tracker = region_config['disable_tracker']
    shift_period = region_config['shift_period']

    # Print data summary
    data = get_data(state=state, district=district, disable_tracker=disable_tracker, use_dataframe='data_all')

    # Output folder
    region = district if district is not None else state
    root_folder = f'{region}/{output_folder}'
    print("Run no. ", num+1, " with shift of ", shift_period * num)
    run_experiments(ihme_config_path, region_config_path, data, f'{root_folder}/{str(num)}', multiple=True,
                    shift_forward=shift_period * num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ihme_config", help="ihme config file name", required=True)
    parser.add_argument("-r", "--region_config", help="region config file name", required=True)
    parser.add_argument("-f", "--output_folder", help="output folder name", required=True)
    parser.add_argument("-n", "--num", help="number of periods to shift forward", required=False, default=1)
    args = parser.parse_args()

    runner(args.ihme_config, args.region_config, args.output_folder, int(args.num))
