import os
import json
import pickle
import pandas as pd

from copy import deepcopy

import yaml

from utils.enums import Columns
from utils.util import read_config
from utils.data import regions

from main.ihme.fitting import single_cycle
from main.seir.fitting import data_setup, run_cycle

from models.seir import SEIR_Testing, SIRD

from viz.forecast import plot_forecast_agnostic
from viz.fit import plot_fit
from viz.synthetic_data import plot_fit_uncertainty


supported_models = [
    {'model': SEIR_Testing, 'name': 'SEIR Testing', 'name_prefix': 'seirt'},
    {'model': SIRD, 'name': 'SIRD', 'name_prefix': 'sird'}
]


def ihme_data_generator(district, state, disable_tracker, actual_start_date, dataset_length,
                        train_val_size, val_size, test_size,
                        config_path, output_folder, which_compartments=Columns.curve_fit_compartments()):
    """Runs IHME model, creates plots and returns results

    Args:
        district (str): name of district
        disable_tracker (bool): If False, data from tracker is not used
        actual_start_date (datetime.datetime): date from which series s1 begins
        dataset_length (int): length of dataset used
        train_val_size (int): length of train+val split
        val_size (int): length of val split (within train+val split)
        test_size (int): length of test split
        config_path (str): path to config file
        output_folder (str): path to output folder
        which_compartments (list, optional): list of compartments to fit (default: Columns.curve_fit_compartments())

    Returns:
        dict, dict, dict: results, updated config, model parameters
    """

    region = district if district is not None else state
    district, state, area_names = regions[region.lower()]
    config, model_params = read_config(config_path)

    config['start_date'] = actual_start_date
    config['dataset_length'] = dataset_length
    config['disable_tracker'] = disable_tracker
    config['test_size'] = test_size
    config['val_size'] = val_size

    ihme_res = single_cycle(district, state, area_names, model_params, which_compartments=which_compartments, **config)

    ihme_df_train, ihme_df_val = ihme_res['df_train'], ihme_res['df_val']
    ihme_df_train_nora, ihme_df_val_nora = ihme_res['df_train_nora'], ihme_res['df_val_nora']
    ihme_df_true = ihme_res['df_district']
    ihme_df_pred = ihme_res['df_prediction']

    makesum = deepcopy(ihme_df_pred)
    if set(which_compartments) == set(Columns.curve_fit_compartments()):
        makesum['total_infected'] = ihme_df_pred['recovered'] + ihme_df_pred['deceased'] + ihme_df_pred['hospitalised']

    ihme_res['df_final_prediction'] = makesum

    plot_fit(
        makesum.reset_index(), ihme_df_train, ihme_df_val, ihme_df_true,
        train_val_size, state, district, which_compartments=[c.name for c in which_compartments],
        description='Train and test',
        savepath=os.path.join(output_folder, 'ihme_i1_fit.png'))

    plot_forecast_agnostic(ihme_df_true, makesum.reset_index(), model_name='IHME',
                           dist=district, state=state, which_compartments=which_compartments,
                           filename=os.path.join(output_folder, 'ihme_i1_forecast.png'))

    for plot_col in which_compartments:
        plot_fit_uncertainty(makesum.reset_index(), ihme_df_train, ihme_df_val, ihme_df_train_nora, ihme_df_val_nora,
                             train_val_size, test_size, state, district, draws=ihme_res['draws'],
                             which_compartments=[plot_col.name],
                             description='Train and test',
                             savepath=os.path.join(output_folder, f'ihme_i1_{plot_col.name}.png'))

    return ihme_res, config, model_params


def seir_runner(district, state, input_df, data_from_tracker,
                train_period, val_period, which_compartments,
                model=SEIR_Testing,  variable_param_ranges=None, num_evals=1500):
    """Wrapper for main.seir.fitting.run_cycle

    Args:
        district (str): name of district
        state (str): name of state
        input_df (tuple): dataframe of observations from districts_daily/custom file/athena DB, dataframe of raw data
        data_from_tracker (bool): If False, data from tracker is not used
        train_period (int): length of train period
        val_period (int): length of val period
        which_compartments (list(enum)): list of compartments to fit on
        model (object, optional): model class to be used (default: SEIR_Testing)
        variable_param_ranges(dict, optional): search space (default: None)
        num_evals (int, optional): number of evaluations of bayesian optimisation (default: 1500)

    Returns:
        dict: results
    """

    observed_dataframes = data_setup(input_df, val_period, continuous_ra=False)
    predictions_dict = run_cycle(
        state, district, observed_dataframes,
        model=model, variable_param_ranges=variable_param_ranges,
        data_from_tracker=data_from_tracker, train_period=train_period,
        which_compartments=which_compartments, N=1e7,
        num_evals=num_evals, initialisation='intermediate'
    )
    return predictions_dict


def log_experiment_local(output_folder, i1_config, i1_model_params, i1_output,
                         c1_output, datasets, predictions_dicts, which_compartments, replace_compartments,
                         dataset_properties, series_properties, baseline_predictions_dict=None, name_prefix="",
                         variable_param_ranges=None):
    """Logs all results

    Args:
        output_folder (str): Output folder path
        i1_config (dict): Config for IHME I1 model
        i1_model_params (dict): Model parameters of IHME I1 model
        i1_output (dict): Results dict of IHME I1 model
        c1_output (dict): Results dict of SEIR C1 model
        datasets (dict): Dictionary of datasets used in experiments
        predictions_dicts (dict): Dictionary of results dicts from SEIR c2 models
        which_compartments (list, optional): list of compartments fitted by SEIR
        replace_compartments (list, optional): list of compartments for which synthetic data is used
        dataset_properties (dict): Properties of datasets used
        series_properties (dict): Properties of series used in experiments
        baseline_predictions_dict (dict, optional): Results dict of SEIR c3 baseline model
        name_prefix (str): prefix for filename
    """

    params_dict = {
        'compartments_replaced': replace_compartments,
        'dataset_properties': {
            'exp' + str(i+1): dataset_properties[i] for i in range(len(dataset_properties))
        },
        'series_properties': series_properties
    }

    for i in datasets:
        filename = 'dataset_experiment_' + str(i+1) + '.csv'
        datasets[i].to_csv(output_folder+filename)

    c1 = c1_output['df_loss'].T[which_compartments]
    c1.to_csv(output_folder + name_prefix + "_c1_loss.csv")

    c1_output['plots']['fit'].savefig(output_folder + name_prefix + '_c1.png')
    c1_output['pointwise_train_loss'].to_csv(output_folder + name_prefix + "_c1_pointwise_train_loss.csv")
    c1_output['pointwise_val_loss'].to_csv(output_folder + name_prefix + "_c1_pointwise_val_loss.csv")

    with open(f'{output_folder}{name_prefix}_c1_best_params.json', 'w') as outfile:
        json.dump(c1_output['best_params'], outfile, indent=4)
    if variable_param_ranges is not None:
        with open(f'{output_folder}{name_prefix}_variable_param_ranges.json', 'w') as outfile:
            json.dump(variable_param_ranges, outfile, indent=4)

    i1 = i1_output['df_loss']
    i1.to_csv(output_folder + "ihme_i1_loss.csv")

    i1_output['df_train_loss_pointwise'].to_csv(output_folder + "ihme_i1_pointwise_train_loss.csv")
    i1_output['df_test_loss_pointwise'].to_csv(output_folder + "ihme_i1_pointwise_test_loss.csv")

    loss_dfs = []
    for i in predictions_dicts:
        loss_df = predictions_dicts[i]['df_loss'].T[which_compartments]
        loss_df['exp'] = i+1
        loss_dfs.append(loss_df)
        predictions_dicts[i]['plots']['fit'].savefig(output_folder + name_prefix + '_c2_experiment_'+str(i+1)+'.png')
        predictions_dicts[i]['pointwise_train_loss'].to_csv(
            output_folder + name_prefix + "_c2_pointwise_train_loss_exp_"+str(i+1)+".csv")
        predictions_dicts[i]['pointwise_val_loss'].to_csv(
            output_folder + name_prefix + "_c2_pointwise_val_loss_exp_"+str(i+1)+".csv")
        with open(f'{output_folder}{name_prefix}_c2_best_params_exp_{str(i+1)}.json', 'w') as outfile:
            json.dump(predictions_dicts[i]['best_params'], outfile, indent=4)

    loss = pd.concat(loss_dfs, axis=0)
    loss.index.name = 'index'
    loss.sort_values(by='index', inplace=True)
    loss.to_csv(output_folder + name_prefix + "_experiments_loss.csv")

    if baseline_predictions_dict is not None:
        baseline_loss = baseline_predictions_dict['df_loss_s3'].T[replace_compartments]
        baseline_loss.to_csv(output_folder + name_prefix + "_baseline_loss.csv")
        with open(f'{output_folder}{name_prefix}_c3_best_params.json', 'w') as outfile:
            json.dump(baseline_predictions_dict['best_params'], outfile, indent=4)
        baseline_predictions_dict['pointwise_train_loss'].to_csv(
            output_folder + name_prefix + "_c3_pointwise_train_loss.csv")
        baseline_predictions_dict['pointwise_val_loss'].to_csv(
            output_folder + name_prefix + "_c3_pointwise_val_loss.csv")

    with open(output_folder + name_prefix + '_experiments_params.json', 'w') as outfile:
        json.dump(params_dict, outfile, indent=4)

    i1_config_dump = deepcopy(i1_config)
    i1_config_dump['start_date'] = i1_config_dump['start_date'].strftime("%Y-%m-%d")
    with open(output_folder + 'ihme_i1_config.json', 'w') as outfile:
        json.dump(i1_config_dump, outfile, indent=4)

    with open(output_folder + 'ihme_i1_model_params.json', 'w') as outfile:
        json.dump(repr(i1_model_params), outfile, indent=4)

    i1_dump = deepcopy(i1_output)
    for var in i1_dump['individual_results']:
        individual_res = i1_dump['individual_results'][var]
        del individual_res['optimiser']

    picklefn = f'{output_folder}/i1.pkl'
    with open(picklefn, 'wb') as pickle_file:
        pickle.dump(i1_dump, pickle_file)

    picklefn = f'{output_folder}/{name_prefix}_c1.pkl'
    with open(picklefn, 'wb') as pickle_file:
        pickle.dump(c1_output, pickle_file)

    picklefn = f'{output_folder}/{name_prefix}_c2.pkl'
    with open(picklefn, 'wb') as pickle_file:
        pickle.dump(predictions_dicts, pickle_file)

    picklefn = f'{output_folder}/{name_prefix}_c3.pkl'
    with open(picklefn, 'wb') as pickle_file:
        pickle.dump(baseline_predictions_dict, pickle_file)


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


def get_variable_param_ranges_dict(district, state, model_type='seirt', train_period=7):
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

    return read_region_params_config(f'../../scripts/ihme_seir/config/{config_name}.yaml', model_type, train_period)


def read_region_config(path):
    """Reads config file for synthetic data generation experiments

    Args:
        path (str): path to config file

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
    config = config['base']
    return config


def read_region_params_config(path, model_type, train_period):
    """Reads config file for variable param ranges for a region

    Args:
        path (str): path to config file
        model_type (str, optional): type of compartmental model [seirt or sird]
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
        config = config[f'params_{model_type}_tp_{train_period}']
    except KeyError:
        config = base_config[f'params_{model_type}_tp_{train_period}']
    return config
