import collections
import itertools
import json
import os
import sys
from abc import ABCMeta
from copy import deepcopy
from datetime import datetime

import numpy as np
from hyperopt import hp

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

sys.path.append('../../')

from utils.generic.enums.columns import Columns


class HidePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def set_variable_param_ranges(variable_param_ranges, fitting_method='bo_hyperopt'):
    """Returns the ranges for the variable params in the search space

    Keyword Arguments:

        as_str {bool} -- If true, the parameters are not returned as a hyperopt object, but as a dict in
        string form (default: {False})

    Returns:
        dict -- dict of ranges of variable params
    """

    formatted_param_ranges = {}
    if fitting_method == 'bo_hyperopt':
        for key in variable_param_ranges.keys():
            formatted_param_ranges[key] = getattr(hp, variable_param_ranges[key][1])(
                key, variable_param_ranges[key][0][0], variable_param_ranges[key][0][1])

    if fitting_method == 'gridsearch':
        for key in variable_param_ranges.keys():
            formatted_param_ranges[key] = np.linspace(variable_param_ranges[key][0][0],
                                                        variable_param_ranges[key][0][1],
                                                        variable_param_ranges[key][1])

    return formatted_param_ranges

def train_test_split(df, threshold, threshold_col='date'):
    return df[df[threshold_col] <= threshold], df[df[threshold_col] > threshold]


def smooth(y, smoothing_window):
    box = np.ones(smoothing_window) / smoothing_window
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def rollingavg(series, window):
    return series.rolling(window).mean()


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
        if isinstance(v, collectionsAbc.MutableMapping):
            new_dict[k] = update_dict(new_dict.get(k, {}), dict(v))
        else:
            new_dict[k] = v
    return new_dict


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.strftime('%m-%d-%Y')
        elif isinstance(obj, ABCMeta):
            return obj.__name__
        elif isinstance(obj, Columns):
            return obj.name
        elif isinstance(obj, type):
            return obj.__name__
        elif callable(obj):
            return obj.__name__
        else:
            return super(CustomEncoder, self).default(obj)


def chunked(iterable, size=1):
    """Divide iterable into chunks of specified size"""
    for it in iterable:
        yield itertools.chain([it], itertools.islice(iterable, size - 1))
