import yaml
import copy

import models
import main.seir.uncertainty as uncertainty_module
from utils.generic.enums import Columns

import datetime


def read_config(filename='default.yaml', preprocess=True, config_dir='seir'):
    """Function for reading the YAML config file and doing some preprocessing

    Args:
        filename (str, optional): Config file name. Defaults to 'default.yaml'.
        preprocess (bool, optional): If False, "processing" such as calling class names 
        inputted by the user, etc are not done. The config is returned as is. 
        This feature is used for W&B. Defaults to True.
        config_dir (str, optional): Config directory name (Default: seir)

    Returns:
        dict: dict of config params
    """
    config_path = f'../../configs/{config_dir}/{filename}'
    with open(config_path) as configfile:
        config = yaml.load(configfile, Loader=yaml.SafeLoader)
    
    if not preprocess:
        return config
    else:
        return process_config(config)

def create_location_description(nconfig):
    """Helper function for creating location description

    Args:
        nconfig (dict): The input config
    """
    dl_nconfig = nconfig['fitting']['data']['dataloading_params']
    if 'state' in dl_nconfig.keys() and 'district' in dl_nconfig.keys():
        location_description = (dl_nconfig['state'], dl_nconfig['district'])
    elif 'region' in dl_nconfig.keys() and 'sub_region' in dl_nconfig.keys():
        location_description = (dl_nconfig['region'], dl_nconfig['sub_region'])
    elif 'state' in dl_nconfig.keys() and 'county' in dl_nconfig.keys():
        location_description = (dl_nconfig['state'], dl_nconfig['county'])
    else:
        location_description = (dl_nconfig['state'])
    dl_nconfig['location_description'] = location_description

def process_config(config):
    """Helper function for processing config file read from yaml file

    Args:
        config (dict): Unprocessed config dict

    Returns:
        dict: Processed config dict
    """
    nconfig = copy.deepcopy(config)
    create_location_description(nconfig)
    
    nconfig['fitting']['model'] = getattr(models.seir, nconfig['fitting']['model'])

    nconfig['forecast']['plot_topk_trials_for_columns'] = [Columns.from_name(
        column) for column in nconfig['forecast']['plot_topk_trials_for_columns']]
    nconfig['forecast']['plot_ptiles_for_columns'] = [Columns.from_name(
        column) for column in nconfig['forecast']['plot_ptiles_for_columns']]

    nconfig['uncertainty']['method'] = getattr(uncertainty_module, nconfig['uncertainty']['method'])
    nconfig['uncertainty']['uncertainty_params']['sort_trials_by_column'] = Columns.from_name(
        nconfig['uncertainty']['uncertainty_params']['sort_trials_by_column'])
        
    return nconfig

def make_date_key_str(config):
    """A function that loops recursively across the input config and 
    converts any element where the key is a datetime.date to an element with a str key
    W&B doesn't allow datetime.date keys in configs

    Args:
        config (dict): input config to fitting

    Returns:
        dict: Config with datetime.date keys converted to str
    """
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
