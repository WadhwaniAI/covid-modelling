import os
import sys
from datetime import datetime

import numpy as np
import yaml
from copy import deepcopy
import collections.abc

class HidePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def train_test_split(df, threshold, threshold_col='date'):
    return df[df[threshold_col] <= threshold], df[df[threshold_col] > threshold]

def smooth(y, smoothing_window):
    box = np.ones(smoothing_window)/smoothing_window
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def rollingavg(series, window):
    return series.rolling(window, center=True).mean()

def read_config(path, backtesting=False):
    default_path = os.path.join(os.path.dirname(path), 'default.yaml')
    with open(default_path) as default:
        config = yaml.load(default, Loader=yaml.SafeLoader)
    with open(path) as configfile:
        new = yaml.load(configfile, Loader=yaml.SafeLoader)
    for k in config.keys():
        if type(config[k]) is dict and new.get(k) is not None:
            config[k].update(new[k])
    model_params = config['model_params']
    if backtesting:
        config['base'].update(config['backtesting'])
    else:
        config['base'].update(config['run'])
    config = config['base']
    return config, model_params

def update_dict(dict_1, dict_2):
    """Update one nested dictionary with another

    Args:
        dict_1 (dict): dictionary which is updated
        dict_2 (dict): dictionary from values are copied

    Returns:
        dict: updated dictionary
    """
    new_dict = deepcopy(dict_1)
    for k, v in dict_2.items():
        if isinstance(v, collections.abc.MutableMapping):
            new_dict[k] = update_dict(new_dict.get(k, {}), dict(v))
        else:
            new_dict[k] = v
    return new_dict

def get_subset(df, lower, upper, col='date'):
    """Get subset of rows of dataframe"""
    lower = lower if lower is not None else df.iloc[0, :][col]
    upper = upper if upper is not None else df.iloc[-1, :][col]
    return df[np.logical_and(df[col] >= lower, df[col] <= upper)]

def convert_date(date, to_str=False, format='%m-%d-%Y'):
    """Convert date between string and datetime.datetime formats

    Args:
        date (Any): date to be converted
        to_str (bool): if True, perform datetime to string conversion, otherwise string to datetime
        format (str): date format

    Returns:
        Any: Converted date
    """
    try:
        if to_str:
            return date.strftime(format)
        else:
            return datetime.strptime(date, format)
    except:
        return date


def read_file(path, file_type='yaml'):
    if file_type == 'yaml':
        with open(path) as infile:
            config = yaml.load(infile, Loader=yaml.SafeLoader)
    return config
