import collections
import copy
import itertools
import json
import pickle
import yaml
import os
import sys
from abc import ABCMeta
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
from hyperopt import hp

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

sys.path.append('../../')

from utils.generic.enums.columns import Columns
from utils.generic.config import make_date_str


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


def get_ensemble_params(params, losses, beta, return_dev=False):
    df_trials = pd.DataFrame.from_dict(params.tolist())
    df_trials['loss'] = losses
    df_trials['loss_wt'] = np.exp(-beta * df_trials['loss'])
    em_params = df_trials.iloc[:, :-2].apply(
        lambda x: np.average(x, weights=df_trials['loss_wt'])).to_dict()

    if return_dev:
        df_mean = pd.DataFrame.from_dict(em_params, orient='index').T
        df_var = copy.deepcopy(df_trials)
        df_var.iloc[:, :-2] = (df_var.iloc[:, :-2] - df_mean.to_numpy())**2

        em_params_dev = df_var.iloc[:, :-2].apply(lambda x: np.sqrt(
            np.average(x, weights=df_var['loss_wt']))).to_dict()

        return em_params, em_params_dev
    else:
        return em_params


def create_output(predictions_dict, output_folder, tag):
    """Custom output generation function"""
    directory = f'{output_folder}/{tag}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    d = {}
    for key in ['variable_param_ranges', 'best_params', 'beta_loss']:
        if key in predictions_dict:
            with open(f'{directory}/{key}.json', 'w') as f:
                json.dump(predictions_dict[key], f, indent=4)
    for key in ['df_prediction', 'df_district', 'df_train', 'df_val', 'df_loss', 'df_district_unsmoothed']:
        if key in predictions_dict and predictions_dict[key] is not None:
            predictions_dict[key].to_csv(f'{directory}/{key}.csv')
    for key in ['trials', 'run_params', 'plots', 'smoothing_description', 'default_params']:
        if key in predictions_dict and predictions_dict[key] is not None:
            with open(f'{directory}/{key}.pkl', 'wb') as f:
                pickle.dump(predictions_dict[key], f)
    if 'ensemble_mean' in predictions_dict['forecasts']:
        predictions_dict['forecasts']['ensemble_mean'].to_csv(
            f'{directory}/ensemble_mean_forecast.csv')
    pd.concat(predictions_dict['trials']['predictions']).to_csv(
        f'{directory}/trials_predictions.csv')
    np.save(f'{directory}/trials_params.npy',
            predictions_dict['trials']['params'])
    np.save(f'{directory}/trials_losses.npy',
            predictions_dict['trials']['losses'])
    for key in ['data_last_date', 'fitting_date']:
        if key in predictions_dict:
            d[key] = predictions_dict[key]
    if len(d) > 0:
        with open(f'{directory}/other.json', 'w') as f:
            json.dump(d, f, indent=4)

    if 'beta' in  predictions_dict:
        np.save(f'{directory}/beta.npy', predictions_dict['beta'])
    try:
        with open(f'{directory}/config.json', 'w') as f:
            json.dump(make_date_str(
                predictions_dict['config']), f, indent=4, cls=CustomEncoder)
        with open(f'{directory}/config.yaml', 'w') as f:
            yaml.dump(make_date_str(predictions_dict['config']), f)
    except Exception:
        pass


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
