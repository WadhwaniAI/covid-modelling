import collections
import os
import sys
from copy import deepcopy

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

import numpy as np


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
