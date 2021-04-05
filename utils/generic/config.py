import copy
import datetime
import importlib
import itertools

import pandas as pd
import yaml

from utils.fitting.util import update_dict
from utils.generic.enums import Columns


def read_config(filename='default.yaml', preprocess=True, config_dir='seir', abs_path=False):
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
    if not abs_path:
        config_path = f'../../configs/{config_dir}/{filename}'
    else:
        config_path = filename
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
    elif 'state' in dl_nconfig.keys():
        location_description = (dl_nconfig['state'])
    else:
        location_description = ''
    dl_nconfig['location_description'] = location_description


def process_config_seir(config):
    """Helper function for processing SEIR config file read from yaml file

    Args:
        config (dict): Unprocessed config dict

    Returns:
        dict: Processed config dict
    """
    nconfig = copy.deepcopy(config)
    create_location_description(nconfig)
    
    model_family = importlib.import_module(
        f".{nconfig['fitting']['model_family']}", 'models')

    nconfig['fitting']['model'] = getattr(model_family, nconfig['fitting']['model'])

    nconfig['plotting']['plot_topk_trials_for_columns'] = [Columns.from_name(
        column) for column in nconfig['plotting']['plot_topk_trials_for_columns']]
    nconfig['plotting']['plot_ptiles_for_columns'] = [Columns.from_name(
        column) for column in nconfig['plotting']['plot_ptiles_for_columns']]

    import main.seir.uncertainty as uncertainty_module
    nconfig['uncertainty']['method'] = getattr(uncertainty_module, nconfig['uncertainty']['method'])
    nconfig['uncertainty']['uncertainty_params']['sort_trials_by_column'] = Columns.from_name(
        nconfig['uncertainty']['uncertainty_params']['sort_trials_by_column'])
        
    return nconfig


def process_config_ihme(config):
    """Helper function for processing IHME config file read from yaml file

    Args:
        config (dict): Unprocessed config dict

    Returns:
        dict: Processed config dict
    """
    from curvefit.core import functions

    nconfig = copy.deepcopy(config)
    create_location_description(nconfig)

    import models
    nconfig['fitting']['model'] = getattr(models.ihme, nconfig['fitting']['model'])
    nconfig['fitting']['model_params']['func'] = getattr(functions, nconfig['fitting']['model_params']['func'])
    nconfig['fitting']['model_params']['covs'] = nconfig['fitting']['data']['covariates']

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


def to_str(obj):
    if isinstance(obj, datetime.datetime):
        return obj.strftime('%Y-%m-%d')
    return str(obj)


def expand_choices(pattern, choices):
    if pattern == 'list':
        pass
    elif pattern == 'repeat':
        choices = [choices[0]] * choices[1]
    elif pattern == 'range':
        choices = range(choices[0], choices[1], choices[2])
    elif pattern == 'date_range':
        choices = {k: v for k, v in zip(['start', 'end', 'periods', 'freq'],
                                        choices) if v is not None}
        choices = pd.date_range(**choices).date
    else:
        raise Exception('Undefined pattern')
    return choices


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
                if pattern == 'string':
                    if len(choices) < 2:
                        raise Exception('Insufficient arguments')
                    else:
                        format_strings = [choices[0]] * choices[1]
                        for i in range(2, len(choices)):
                            choice_list = expand_choices(choices[i][0], choices[i][1])
                            format_strings = [fs.replace('<>', to_str(c)) for fs, c in
                                              zip(format_strings, choice_list)]
                        choices = format_strings
                else:
                    choices = expand_choices(pattern, choices)
                new_config[k] = choices
    return new_config


def split_dict_combinations(d):
    """Convert dict of lists into list of dicts of all combinations"""
    keys, values = d.keys(), d.values()
    values_choices = (split_dict_combinations(v) if isinstance(v, dict) else v for v in values)
    for comb in itertools.product(*values_choices):
        yield dict(zip(keys, comb))


def split_dict_vertical(d):
    """Convert dict of lists to list of dicts"""
    keys, values = d.keys(), d.values()
    values_choices = (split_dict_vertical(v) if isinstance(v, dict) else v for v in values)
    for comb in zip(*values_choices):
        yield dict(zip(keys, comb))


def chain(keys, iterables):
    for i, it in enumerate(iterables):
        for element in it:
            yield keys[i], element


def generators_product(g1, g2):
    for (c1, c2) in itertools.product(g1, g2):
        yield update_dict(c1, c2)


def generate_configs_from_driver(driver_config_filename):
    driver_config = read_config(driver_config_filename, preprocess=False, config_dir='exper')
    if 'constant' in driver_config and driver_config['constant'] is not None:
        configs = []
        for config_name in driver_config['configs']:
            base_configs = generate_config(driver_config['iterate'])
            base_configs = split_dict_combinations(base_configs)
            if config_name in driver_config['constant']:
                temp = generate_config(driver_config['constant'][config_name])
                if temp == {}:
                    config = base_configs
                else:
                    specific_config = split_dict_vertical(temp)
                    config = generators_product(base_configs, specific_config)
            else:
                config = base_configs
            configs.append(config)
        configs = chain(driver_config['configs'], configs)
    else:
        base_configs = generate_config(driver_config['iterate'])
        configs = chain(driver_config['configs'], (split_dict_combinations(base_configs)
                                                   for _ in driver_config['configs']))
    return configs


if __name__ == '__main__':
    configs = generate_configs_from_driver('list_of_exp.yaml')
