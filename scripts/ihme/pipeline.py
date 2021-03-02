"""
pipeline.py
"""
import argparse
import copy
import datetime

from main.ihme.fitting import single_fitting_cycle
from utils.generic.config import read_config


def run_pipeline(config_filename):
    config = read_config(config_filename, preprocess=False)
    timestamp = datetime.datetime.now()
    output_folder = '../../misc/ihme/{}'.format(timestamp.strftime("%Y_%m%d_%H%M%S"))
    predictions_dict = single_fitting_cycle(**copy.deepcopy(config['fitting']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config file name", required=True)
    args = parser.parse_args()

    run_pipeline(args.config)
