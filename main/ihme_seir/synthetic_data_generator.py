import os
import json
import pickle
import pandas as pd

from copy import deepcopy

from utils.enums import Columns
from utils.util import read_config
from utils.data import cities

from main.ihme.fitting import single_cycle
from main.seir.fitting import data_setup, run_cycle

from models.seir import SEIR_Testing, SIRD

from viz.forecast import plot_forecast_agnostic
from viz.fit import plot_fit
from viz.synthetic_data import plot_fit_uncertainty


def ihme_data_generator(district, disable_tracker, actual_start_date, dataset_length,
                        train_val_size, val_size, test_size,
                        config_path, output_folder):
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

    Returns:
        dict, dict, dict: results, updated config, model parameters
    """

    district, state, area_names = cities[district.lower()]  # FIXME
    config, model_params = read_config(config_path)

    config['start_date'] = actual_start_date
    config['dataset_length'] = dataset_length
    config['disable_tracker'] = disable_tracker
    config['test_size'] = test_size
    config['val_size'] = val_size

    ihme_res = single_cycle(district, state, area_names, model_params, **config)

    ihme_df_train, ihme_df_val = ihme_res['df_train'], ihme_res['df_val']
    ihme_df_train_nora, ihme_df_val_nora = ihme_res['df_train_nora'], ihme_res['df_val_nora']
    ihme_df_true = ihme_res['df_district']
    ihme_df_pred = ihme_res['df_prediction']

    makesum = deepcopy(ihme_df_pred)
    makesum['total_infected'] = ihme_df_pred['recovered'] + ihme_df_pred['deceased'] + ihme_df_pred['hospitalised']
    ihme_res['df_final_prediction'] = makesum

    plot_fit(
        makesum.reset_index(), ihme_df_train, ihme_df_val, ihme_df_train_nora, ihme_df_val_nora,
        train_val_size, state, district, which_compartments=[c.name for c in Columns.curve_fit_compartments()],
        description='Train and test',
        savepath=os.path.join(output_folder, 'ihme_i1_fit.png'))

    plot_forecast_agnostic(ihme_df_true, makesum.reset_index(), model_name='IHME',
                           dist=district, state=state, filename=os.path.join(output_folder, 'ihme_i1_forecast.png'))

    for plot_col in Columns.curve_fit_compartments():
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

    predictions_dict = dict()
    observed_dataframes = data_setup(input_df[0], input_df[1], val_period, continuous_ra=False)
    predictions_dict['m1'] = run_cycle(
        state, district, observed_dataframes,
        model=model, variable_param_ranges=variable_param_ranges,
        data_from_tracker=data_from_tracker, train_period=train_period,
        which_compartments=which_compartments, N=1e7,
        num_evals=num_evals, initialisation='intermediate'
    )
    return predictions_dict


def log_experiment_local(output_folder, i1_config, i1_model_params, i1_output,
                         c1_output, datasets, predictions_dicts, which_compartments,
                         dataset_properties, series_properties, baseline_predictions_dict=None, name_prefix=""):
    """Logs all results

    Args:
        output_folder (str): Output folder path
        i1_config (dict): Config for IHME I1 model
        i1_model_params (dict): Model parameters of IHME I1 model
        i1_output (dict): Results dict of IHME I1 model
        c1_output (dict): Results dict of SEIR C1 model
        datasets (dict): Dictionary of datasets used in experiments
        predictions_dicts (dict): Dictionary of results dicts from SEIR c2 models
        which_compartments (list(enum), optional): list of compartments for which synthetic data is used
        dataset_properties (dict): Properties of datasets used
        series_properties (dict): Properties of series used in experiments
        baseline_predictions_dict (dict, optional): Results dict of SEIR c3 baseline model
        name_prefix (str): prefix for filename
    """

    params_dict = {
        'compartments_replaced': which_compartments,
        'dataset_properties': {
            'exp' + str(i+1): dataset_properties[i] for i in range(len(dataset_properties))
        },
        'series_properties': series_properties
    }

    for i in datasets:
        filename = 'dataset_experiment_' + str(i+1) + '.csv'
        datasets[i].to_csv(output_folder+filename)

    c1 = c1_output['m1']['df_loss'].T[which_compartments]
    c1.to_csv(output_folder + name_prefix + "_c1_loss.csv")

    c1_output['m1']['ax'].savefig(output_folder + name_prefix + '_c1.png')
    c1_output['m1']['pointwise_train_loss'].to_csv(output_folder + name_prefix + "_c1_pointwise_val_loss.csv")
    c1_output['m1']['pointwise_val_loss'].to_csv(output_folder + name_prefix + "_c1_pointwise_val_loss.csv")

    i1 = i1_output['df_loss']
    i1.to_csv(output_folder + "ihme_i1_loss.csv")

    i1_output['df_train_loss_pointwise'].to_csv(output_folder + "ihme_i1_pointwise_train_loss.csv")
    i1_output['df_test_loss_pointwise'].to_csv(output_folder + "ihme_i1_pointwise_test_loss.csv")

    loss_dfs = []
    for i in predictions_dicts:
        loss_df = predictions_dicts[i]['m1']['df_loss'].T[which_compartments]
        loss_df['exp'] = i+1
        loss_dfs.append(loss_df)
        predictions_dicts[i]['m1']['ax'].savefig(output_folder + name_prefix + '_c2_experiment_'+str(i+1)+'.png')
        predictions_dicts[i]['m1']['pointwise_train_loss'].to_csv(
            output_folder + name_prefix + "_c2_pointwise_train_loss_exp_"+str(i+1)+".csv")
        predictions_dicts[i]['m1']['pointwise_val_loss'].to_csv(
            output_folder + name_prefix + "_c2_pointwise_val_loss_exp_"+str(i+1)+".csv")

    loss = pd.concat(loss_dfs, axis=0)
    loss.index.name = 'index'
    loss.sort_values(by='index', inplace=True)
    loss.to_csv(output_folder + name_prefix + "_experiments_loss.csv")

    if baseline_predictions_dict is not None:
        baseline_loss = baseline_predictions_dict['m1']['df_loss_s3'].T[which_compartments]
        baseline_loss.to_csv(output_folder + name_prefix + "_baseline_loss.csv")

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


def get_variable_param_ranges_dict(model):
    if model is SEIR_Testing:
        print("Getting search space for:", model)
        return None
    elif model is SIRD:
        print("Getting search space for:", model)
        return {
            'lockdown_R0': (1, 6),
            'T_inc': (4, 16),
            'T_inf': (10, 60),
            'T_fatal': (200, 300)
        }
    else:
        raise Exception("This model class is not supported")
