"""
Script to run IHME-SEIR synthetic data experiments. Experiments are run sequentially.
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

Usage example: python3 synthetic_data_experiments_seq.py -i ../ihme/config/pune.yaml -r config/pune.yaml -n 10 -s 5
"""

import argparse
import sys
import time

from datetime import datetime

sys.path.append('../../')

from main.ihme_seir.experiments import run_experiments

from data.processing import get_data_from_source

from main.ihme_seir.utils import read_region_config


def run_experiments_over_time(ihme_config_path, region_config_path, num, shift):
    """Performs multiple runs of experiments for a region

    Args:
        ihme_config_path (str): path to ihme config
        region_config_path (str): path to experiment config for region
        num (int): number of runs
        shift (int): number of days to shift forward in each run to get start date of the run

    """

    region_config = read_region_config(region_config_path)
    sub_region = region_config['sub_region']
    region = region_config['region']
    data_source = region_config['data_source']

    # Print data summary
    data = get_data_from_source(region=region, sub_region=sub_region, data_source=data_source)
    print("Data summary:")
    print(data)

    # Output folder
    now = datetime.now()
    date_now = now.strftime("%Y%m%d")
    time_now = now.strftime("%H%M%S")
    region_name = sub_region if sub_region is not None else region
    root_folder = f'{region_name}/{date_now}/{time_now}'

    if num == 1:
        start_time = time.time()
        run_experiments(ihme_config_path, region_config_path, data, root_folder, multiple=False,
                        shift_forward=0)
        runtime = time.time() - start_time
        print("Run time: ", runtime)
    else:
        for i in range(num):
            start_time = time.time()
            print("Run no. ", i + 1)
            run_experiments(ihme_config_path, region_config_path, data, f'{root_folder}/{str(i)}', multiple=True,
                            shift_forward=shift * i)
            runtime = time.time() - start_time
            print("Run time: ", runtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ihme_config", help="ihme config file name", required=True)
    parser.add_argument("-r", "--region_config", help="region config file name", required=True)
    parser.add_argument("-n", "--num", help="number of times experiments are run", required=False, default=1)
    parser.add_argument("-s", "--shift", help="number of days to shift forward", required=False, default=5)
    args = parser.parse_args()

    run_experiments_over_time(args.ihme_config, args.region_config, int(args.num), int(args.shift))
