import collections
import itertools
import json
import os
import sys
from abc import ABCMeta
from copy import deepcopy
from datetime import datetime

import numpy as np
import yaml

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


def rollingavg(series, window):
    return series.rolling(window, center=True).mean()


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


def get_subset(df, lower, upper, col='date'):
    """Get subset of rows of dataframe"""
    lower = lower if lower is not None else df.iloc[0, :][col]
    upper = upper if upper is not None else df.iloc[-1, :][col]
    return df[np.logical_and(df[col] >= lower, df[col] <= upper)]


def convert_date(date, to_str=False, date_format='%m-%d-%Y'):
    """Convert date between string and datetime.datetime formats

    Args:
        date (Any): date to be converted
        to_str (bool): if True, perform datetime to string conversion, otherwise string to datetime
        date_format (str): date format

    Returns:
        Any: Converted date
    """
    try:
        if to_str:
            return date.strftime(date_format)
        else:
            return datetime.strptime(date, date_format)
    except:
        return date


def read_file(path, file_type='yaml'):
    if file_type == 'yaml':
        with open(path) as infile:
            config = yaml.load(infile, Loader=yaml.SafeLoader)
    return config


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
        else:
            return super(CustomEncoder, self).default(obj)


def chunked(iterable, size=1):
    """Divide iterable into chunks of specified size
    https://stackoverflow.com/questions/24527006/split-a-generator-into-chunks-without-pre-walking-it

    Args:
        iterable ():
        size ():

    Returns:

    """
    iterator = iterable
    for first in iterator:
        yield itertools.chain([first], itertools.islice(iterator, size - 1))
