import copy
import datetime
import itertools
import pandas as pd

import yaml
from curvefit.core import functions

import main.seir.uncertainty as uncertainty_module
import models
from utils.fitting.util import update_dict
from utils.generic.enums import Columns


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
        if config_dir == 'seir':
            return process_config_seir(config)
        elif config_dir == 'ihme':
            return process_config_ihme(config)


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


def process_config_seir(config):
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


def process_config_ihme(config):
    """Helper function for processing config file read from yaml file

    Args:
        config (dict): Unprocessed config dict

    Returns:
        dict: Processed config dict
    """
    nconfig = copy.deepcopy(config)
    create_location_description(nconfig)

    nconfig['fitting']['model'] = getattr(models.ihme, nconfig['fitting']['model'])
    nconfig['model_params']['func'] = getattr(functions, nconfig['model_params']['func'])
    nconfig['model_params']['covs'] = nconfig['fitting']['data']['covariates']

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


def make_date_str(config):
    """A function that loops recursively across the input config and
    converts any element where the key or value is a datetime.date to an element with a str key or value
    Allows conversion to JSON

    Args:
        config (dict): input config to fitting

    Returns:
        dict: Config with datetime.date keys and values converted to str
    """
    keys_to_make_str = []
    new_config = copy.copy(config)
    for k, v in config.items():
        if isinstance(v, dict):
            new_config[k] = make_date_str(v)
        else:
            if isinstance(v, list):
                for i in range(len(v)):
                    if isinstance(v[i], datetime.date):
                        v[i] = v[i].strftime('%Y-%m-%d')
            if isinstance(k, datetime.date):
                keys_to_make_str.append(k)
            if isinstance(v, datetime.date):
                new_config[k] = v.strftime('%Y-%m-%d')

    for k, v in config.items():
        if k in keys_to_make_str:
            new_config[k.strftime('%Y-%m-%d')] = v
            del new_config[k]

    return new_config


def generate_config(config):
    """Generate configuration from template

    Args:
        config ():

    Returns:

    """
    new_config = {}
    for k, v in config.items():
        if isinstance(v, dict):
            temp = generate_config(v)
            if temp:
                new_config[k] = temp
        elif isinstance(v, list):
            pattern, choices, select = v
            if select:
                if pattern == 'repeat':
                    choices = [choices[0]] * choices[1]
                if pattern == 'range':
                    choices = range(choices[0], choices[1], choices[2])
                if pattern == 'date_range':
                    choices = {k: v for k, v in zip(['start', 'end', 'periods', 'freq'], choices) if v is not None}
                    choices = pd.date_range(**choices).date
                new_config[k] = choices
    return new_config


def generate_combinations(d):
    keys, values = d.keys(), d.values()
    values_choices = (generate_combinations(v) if isinstance(v, dict) else v for v in values)
    for comb in itertools.product(*values_choices):
        yield dict(zip(keys, comb))


def chain(keys, iterables):
    for i, it in enumerate(iterables):
        for element in it:
            yield keys[i], element


def get_configs_from_driver(driver_config_filename):
    driver_config = read_config(driver_config_filename, preprocess=False, config_dir='other')
    configs = generate_config(driver_config['base'])
    if driver_config['specific']:
        configs = [update_dict(configs, generate_config(driver_config['specific'][config_name]))
                   if config_name in driver_config['specific'] else configs
                   for config_name in driver_config['configs']]
    configs = chain(driver_config['configs'], (generate_combinations(exp) for exp in configs))
    return configs
