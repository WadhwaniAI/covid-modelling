"""
pipeline.py
"""
import argparse
import copy
import datetime
import sys
import warnings
import pandas as pd

sys.path.append('../../')

from main.ihme.fitting import single_fitting_cycle
from utils.generic.config import read_config

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', module='pandas', category=RuntimeWarning)
warnings.filterwarnings('ignore', module='curvefit', category=RuntimeWarning)
warnings.filterwarnings('ignore', module='numpy', category=RuntimeWarning)


def run_pipeline(config_filename):
    """

    Args:
        config_filename ():

    Returns:

    """
    config = read_config(config_filename, preprocess=True, config_dir='ihme')
    timestamp = datetime.datetime.now()
    output_folder = '../../misc/ihme/{}'.format(timestamp.strftime("%Y_%m%d_%H%M%S"))
    predictions_dict = single_fitting_cycle(**copy.deepcopy(config['fitting']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config file name", required=True)
    args = parser.parse_args()

    run_pipeline(args.config)
