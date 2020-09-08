import yaml
import os

import models
import main.seir.uncertainty as uncertainty_module
from utils.generic.enums import Columns


def read_config(filename='default.yaml'):
    with open(f'../../configs/seir/{filename}') as configfile:
        config = yaml.load(configfile, Loader=yaml.SafeLoader)

    config['fitting']['model'] = getattr(models.seir, config['fitting']['model'])

    config['forecast']['plot_topk_trials_for_columns'] = [Columns.from_name(
        column) for column in config['forecast']['plot_topk_trials_for_columns']]
    config['forecast']['plot_ptiles_for_columns'] = [Columns.from_name(
        column) for column in config['forecast']['plot_ptiles_for_columns']]

    config['uncertainty']['method'] = getattr(uncertainty_module, config['uncertainty']['method'])
    config['uncertainty']['uncertainty_params']['sort_trials_by_column'] = Columns.from_name(
        config['uncertainty']['uncertainty_params']['sort_trials_by_column'])
        
    return config