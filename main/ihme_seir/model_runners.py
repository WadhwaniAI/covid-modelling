import json
import os
import pickle
from copy import deepcopy

import pandas as pd

from data.processing.processing import get_observed_dataframes
from main.ihme.fitting import single_cycle
from main.ihme_seir.utils import read_params_file
from main.seir.fitting import run_cycle
from models.seir import SEIR_Testing
from utils.data import get_supported_regions
from utils.enums import Columns
from utils.util import read_config
from viz.fit import plot_fit
from viz.forecast import plot_forecast_agnostic
from viz.synthetic_data import plot_fit_uncertainty


def ihme_runner(sub_region, region, start_date, dataset_length, train_val_size, val_size, test_size, config_path,
                output_folder, variant='log_erf', model_priors=None, data_source='covid19india',
                which_compartments=Columns.curve_fit_compartments()):
    """Runs IHME model, creates plots and returns results

    Args:
        sub_region (str): name of sub-region (province/state/district)
        region (str): name of region (country/state)
        start_date (datetime.datetime): date from which series s1 begins
        dataset_length (int): length of dataset used
        train_val_size (int): length of train+val split
        val_size (int): length of val split (within train+val split)
        test_size (int): length of test split
        config_path (str): path to config file
        output_folder (str): path to output folder
        variant (str): curve fitting function used
        data_source (str): type of data used
        which_compartments (list, optional): list of compartments to fit (default: Columns.curve_fit_compartments())

    Returns:
        dict, dict, dict: results, updated config, model parameters
    """

    region_name = sub_region if sub_region is not None else region
    region_dict = get_supported_regions()[region_name.lower()]
    sub_region, region, area_names = region_dict['sub_region'], region_dict['region'], region_dict['area_names']
    config, model_params = read_config(config_path)

    config['start_date'] = start_date
    config['dataset_length'] = dataset_length
    config['test_size'] = test_size
    config['val_size'] = val_size
    config['data_source'] = data_source
    params_csv_path = config['params_csv']
    model_params['func'] = variant
    if variant == 'log_expit':  # Predictive validity only supported by Gaussian family of functions
        model_params['pipeline_args']['n_draws'] = 0
    if model_priors is not None:
        model_params['priors'] = model_priors

    if config['params_from_csv']:
        param_ranges = read_params_file(params_csv_path, start_date)
        model_params['priors']['fe_bounds'] = [param_ranges['alpha'], param_ranges['beta'], param_ranges['p']]

    ihme_res = single_cycle(sub_region, region, area_names, model_params, which_compartments=which_compartments,
                            **config)

    predictions = deepcopy(ihme_res['df_prediction'])
    # If fitting to all compartments, constrain total_infected as sum of other compartments
    if set(which_compartments) == set(Columns.curve_fit_compartments()):
        predictions['total_infected'] = ihme_res['df_prediction']['recovered'] + ihme_res['df_prediction']['deceased'] \
                                        + ihme_res['df_prediction']['hospitalised']

    ihme_res['df_final_prediction'] = predictions

    plot_fit(
        predictions.reset_index(), ihme_res['df_train'], ihme_res['df_val'], ihme_res['df_district'],
        train_val_size, region, sub_region, which_compartments=[c.name for c in which_compartments],
        description='Train and test',
        savepath=os.path.join(output_folder, f'ihme_i1_fit_{variant}.png'))

    plot_forecast_agnostic(ihme_res['df_district'], predictions.reset_index(), model_name='IHME',
                           dist=sub_region, state=region, which_compartments=which_compartments,
                           filename=os.path.join(output_folder, f'ihme_i1_forecast_{variant}.png'))

    for plot_col in which_compartments:
        plot_fit_uncertainty(predictions.reset_index(), ihme_res['df_train'], ihme_res['df_val'],
                             ihme_res['df_train_nora'],
                             ihme_res['df_val_nora'], train_val_size, test_size, region, sub_region, uncertainty=False,
                             draws=ihme_res['draws'], which_compartments=[plot_col.name], description='Train and test',
                             savepath=os.path.join(output_folder, f'ihme_i1_{plot_col.name}_{variant}.png'))

    model_params['func'] = variant

    return ihme_res, config, model_params


def seir_runner(sub_region, region, input_df, train_period, val_period, which_compartments, model=SEIR_Testing,
                variable_param_ranges=None, num_evals=1500, N=7e6):
    """Wrapper for main.seir.fitting.run_cycle

    Args:
        sub_region (str): name of district
        region (str): name of state
        input_df (tuple): dataframe of observations from districts_daily/custom file/athena DB, dataframe of raw data
        train_period (int): length of train period
        val_period (int): length of val period
        which_compartments (list(enum)): list of compartments to fit on
        model (object, optional): model class to be used (default: SEIR_Testing)
        variable_param_ranges(dict, optional): search space (default: None)
        num_evals (int, optional): number of evaluations of bayesian optimisation (default: 1500)

    Returns:
        dict: results
    """
    observed_dataframes = get_observed_dataframes(input_df, val_period, 0, which_columns=which_compartments)
    predictions_dict = run_cycle(region, sub_region, observed_dataframes, model=model,
                                 variable_param_ranges=variable_param_ranges, train_period=train_period,
                                 which_compartments=which_compartments, num_evals=num_evals, N=N,
                                 initialisation='intermediate')
    return predictions_dict


def log_experiment_local(output_folder, region_config, i1_config, i1_model_params, i1_output, c1_output, datasets,
                         predictions_dicts, which_compartments, replace_compartments, dataset_properties,
                         series_properties, baseline_predictions_dict=None, name_prefix="", variable_param_ranges=None):
    """Logs all results

    Args:
        output_folder (str): Output folder path
        region_config (dict): Config for experiments
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
        variable_param_ranges(dict): parameter search spaces
    """

    params_dict = {
        'compartments_replaced': replace_compartments,
        'dataset_properties': {
            'exp' + str(i + 1): dataset_properties[i] for i in range(len(dataset_properties))
        },
        'series_properties': series_properties
    }

    with open(output_folder + 'region_config.json', 'w') as outfile:
        json.dump(region_config, outfile, indent=4)

    for i in datasets:
        filename = 'dataset_experiment_' + str(i + 1) + '.csv'
        datasets[i].to_csv(output_folder + filename)

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
        loss_df['exp'] = i + 1
        loss_dfs.append(loss_df)
        predictions_dicts[i]['plots']['fit'].savefig(
            output_folder + name_prefix + '_c2_experiment_' + str(i + 1) + '.png')
        predictions_dicts[i]['pointwise_train_loss'].to_csv(
            output_folder + name_prefix + "_c2_pointwise_train_loss_exp_" + str(i + 1) + ".csv")
        predictions_dicts[i]['pointwise_val_loss'].to_csv(
            output_folder + name_prefix + "_c2_pointwise_val_loss_exp_" + str(i + 1) + ".csv")
        with open(f'{output_folder}{name_prefix}_c2_best_params_exp_{str(i + 1)}.json', 'w') as outfile:
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
        json.dump(i1_model_params, outfile, indent=4)

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
