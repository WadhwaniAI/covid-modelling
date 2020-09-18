import os

import pandas as pd
import yaml

from models.seir import SEIR_Testing, SIRD, SEIRHD, SIR
from utils.util import update_dict


def read_config(path, primary_config_name='base'):
    default_path = os.path.join(os.path.dirname(path), f'{primary_config_name}.yaml')
    with open(default_path) as infile:
        primary_config = yaml.load(infile, Loader=yaml.SafeLoader)
    with open(path) as infile:
        secondary_config = yaml.load(infile, Loader=yaml.SafeLoader)
    primary_config = update_dict(primary_config, secondary_config)
    return primary_config


def read_params_file(params_csv_path, start_date):
    params_df = pd.read_csv(f'config/{params_csv_path}', index_col=0, dayfirst=True)
    params_df.index = pd.to_datetime(params_df.index)
    param_ranges = dict()
    model_params = list(set([x.replace('_low', '').replace('_high', '') for x in params_df.columns]))
    for col in model_params:
        param_ranges[col] = [params_df.loc[start_date, :][f'{col}_low'], params_df.loc[start_date, :][f'{col}_high']]
    return param_ranges  # TODO: CHECK FOR SIR


def get_seir_pointwise_loss_dict(path, file, start=0, end=0):
    loss_dict = dict()
    for i in range(start, end + 1):
        loss_dict[i] = pd.read_csv(f'{path}/{str(i)}/{file}', index_col=['compartment', 'loss_function'])
    return loss_dict


def get_seir_pointwise_loss(loss_dict, compartment, loss_fn):
    losses = []
    for i in loss_dict:
        losses.append(loss_dict[i].loc[(compartment, loss_fn)])
    return losses


def get_seir_loss_dict(path, file, start=0, end=0):
    loss_dict = dict()
    for i in range(start, end + 1):
        loss_dict[i] = pd.read_csv(f'{path}/{str(i)}/{file}', index_col=0)
    return loss_dict


def get_seir_loss(loss_dict, compartment, split):
    losses = []
    for i in loss_dict:
        losses.append(loss_dict[i].loc[split][compartment])
    return losses


def get_ihme_loss_dict(path, file, start=0, end=0):
    loss_dict = dict()
    for i in range(start, end + 1):
        loss_dict[i] = pd.read_csv(f'{path}/{str(i)}/{file}',
                                   index_col=['compartment', 'split', 'loss_function'])
    return loss_dict


def get_ihme_loss(loss_dict, compartment, split, loss_fn):
    losses = []
    for i in loss_dict:
        losses.append(loss_dict[i].loc[(compartment, split, loss_fn)]['loss'])
    return losses


def get_ihme_pointwise_loss(loss_dict, compartment, split, loss_fn):
    losses = []
    for i in loss_dict:
        losses.append(loss_dict[i].loc[(compartment, split, loss_fn)])
    return losses


def create_pointwise_loss_csv_old(path, val_loss, val_period, model, compartment, start, end, outfile='test_loss'):
    val_loss = pd.concat(val_loss).to_frame()
    run_num = [i for i in range(start, end + 1) for j in range(val_period)]
    lookahead = [j for i in range(start, end + 1) for j in range(1, val_period+1)]
    val_loss.insert(0, column='run', value=run_num)
    val_loss.insert(1, column='lookahead', value=lookahead)
    val_loss.to_csv(f'{path}/consolidated/{model}_{compartment}_{outfile}.csv')


def create_pointwise_loss_csv(path, val_loss, model, compartment, start, end, outfile='test_loss'):
    for i in range(start, end+1):
        val_loss[i] = val_loss[i].to_frame()
        val_loss[i].insert(0, column='run', value=i)
        val_loss[i].insert(1, column='lookahead', value=range(1, len(val_loss[i])+1))
    val_loss = pd.concat(val_loss)
    val_loss.to_csv(f'{path}/consolidated/{model}_{compartment}_{outfile}.csv')


def create_output_folder(fname):
    """Creates folder in outputs/ihme_seir/

    Args:
        fname (str): name of folder within outputs/ihme_seir/

    Returns:
        str: output_folder path
    """

    output_folder = f'../../outputs/ihme_seir/{fname}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder


def get_variable_param_ranges_dict(district, state, model_type='seirt', train_period=None):
    """Gets dictionary of variable param ranges from config file

    Args:
        district (str): name of district
        state (str): name of state
        model_type (str, optional): type of compartmental model [seirt or sird]
        train_period (int): length of training period

    Returns:
        dict: variable param ranges
    """
    if district is None:
        config_name = state.lower()
    else:
        config_name = district.lower().replace(" ", "_")

    return read_region_params_config(f'../../scripts/ihme_seir/config/{config_name}.yaml', model_type,
                                     train_period)


def read_region_config(path, key='base'):
    """Reads config file for synthetic data generation experiments

    Args:
        path (str): path to config file
        key (str): nested dict to return

    Returns:
        dict: config for synthetic data generation experiments
    """

    default_path = os.path.join(os.path.dirname(path), 'base.yaml')
    with open(default_path) as base:
        config = yaml.load(base, Loader=yaml.SafeLoader)
    with open(path) as configfile:
        region_config = yaml.load(configfile, Loader=yaml.SafeLoader)
    for k in config.keys():
        if type(config[k]) is dict and region_config.get(k) is not None:
            config[k].update(region_config[k])
    config = config[key]
    return config


def read_region_params_config(path, model_type, train_period=None):
    """Reads config file for variable param ranges for a region

    Args:
        path (str): path to config file
        model_type (str, optional): type of compartmental model [seirt, sird or seirhd]
        train_period (int): length of training period

    Returns:
        dict: variable param ranges for region
    """
    config = None
    try:
        with open(path) as configfile:
            config = yaml.load(configfile, Loader=yaml.SafeLoader)
    except OSError:
        pass

    path = f'../../scripts/ihme_seir/config/base.yaml'
    with open(path) as configfile:
        base_config = yaml.load(configfile, Loader=yaml.SafeLoader)

    try:
        if train_period is None:
            config = config[f'params_{model_type}']
        else:
            config = config[f'params_{model_type}_tp_{train_period}']
    except KeyError:
        config = base_config[f'params_{model_type}_tp_{train_period}']
    return config


def get_model(val):
    for model in supported_models:
        if val == model['name'] or val == model['name_prefix']:
            return model['model']


supported_models = [
    {'model': SEIR_Testing, 'name': 'SEIR Testing', 'name_prefix': 'seirt'},
    {'model': SIRD, 'name': 'SIRD', 'name_prefix': 'sird'},
    {'model': SEIRHD, 'name': 'SEIRHD', 'name_prefix': 'seirhd'},
    {'model': SIR, 'name': 'SIR', 'name_prefix': 'sir'}
]