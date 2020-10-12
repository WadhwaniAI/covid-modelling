import yaml
import os
import copy

import models
import main.seir.uncertainty as uncertainty_module
from utils.generic.enums import Columns

import datetime


def read_config(filename='default.yaml', preprocess=True):
    with open(f'../../configs/seir/{filename}') as configfile:
        config = yaml.load(configfile, Loader=yaml.SafeLoader)
    
    if not preprocess:
        return config
    
    config['fitting']['model'] = getattr(models.seir, config['fitting']['model'])

    config['forecast']['plot_topk_trials_for_columns'] = [Columns.from_name(
        column) for column in config['forecast']['plot_topk_trials_for_columns']]
    config['forecast']['plot_ptiles_for_columns'] = [Columns.from_name(
        column) for column in config['forecast']['plot_ptiles_for_columns']]

    config['uncertainty']['method'] = getattr(uncertainty_module, config['uncertainty']['method'])
    config['uncertainty']['uncertainty_params']['sort_trials_by_column'] = Columns.from_name(
        config['uncertainty']['uncertainty_params']['sort_trials_by_column'])
        
    return config

def make_date_key_str(config):
    keys_to_make_str = []
    new_config = copy.copy(config)
    for k, v in config.items():
        if isinstance(v, dict):
            new_config[k] = make_date_key_str(v)
        else:
            if isinstance(k, datetime.date):
                keys_to_make_str.append(k)

    for k, v in config.items():
        if k in keys_to_make_str:
            new_config[k.strftime('%Y-%m-%d')] = v
            del new_config[k]
    
    return new_config
