"""
Script to compare variants of the IHME model fit using the erf, derf and expit functions.
"""
import argparse
import json
import os
import pickle
import sys
from copy import deepcopy
from datetime import timedelta, datetime

import pandas as pd

sys.path.append('../../')

from main.ihme.fitting import single_cycle
from main.ihme_seir.utils import get_ihme_pointwise_loss, get_ihme_loss_dict, read_config, read_params_file, \
    create_pointwise_loss_csv, create_output_folder
from utils.data import get_supported_regions
from utils.enums import Columns
from utils.util import convert_date
from viz import plot_fit
from viz.forecast import plot_forecast_agnostic


def run_experiments(config_path, output_folder, num):
    config = read_config(config_path, primary_config_name='default')
    base = config['base']
    base.update(config['run'])
    model_params = config['model_params']

    # Unpack parameters from config
    sub_region = base['sub_region']
    region = base['region']
    experiment_name = base['experiment_name']
    which_compartments = base['compartments']
    models = base['models']
    train_val_period = base['train_size']
    test_period = base['test_size']
    data_length = train_val_period + test_period
    shift = base['shift']
    start_date = datetime.strptime(base['start_date'], '%m-%d-%Y') + timedelta(shift * num)
    if start_date + timedelta(data_length) > datetime.today():
        raise Exception('Insufficient data available')
    base['start_date'] = convert_date(start_date, to_str=True, format='%m-%d-%Y')
    params_csv_path = base['params_csv']
    verbose = base['verbose']
    base['data_length'] = train_val_period + test_period

    # Create output folder
    region_name = sub_region if sub_region is not None else region
    root_folder = f'{region_name}/{output_folder}/{str(num)}'
    print(region_name, ": Run no. ", num + 1, " with shift of ", shift * num)
    output_folder = create_output_folder(f'{experiment_name}/{root_folder}/')

    # Enum version of compartments to fit to
    which_compartments_enum = [Columns.from_name(comp) for comp in which_compartments]

    if verbose:
        print("Training starts on:", start_date)

    region_name = sub_region if sub_region is not None else region
    region_dict = get_supported_regions()[region_name.lower()]
    sub_region, region, area_names = region_dict['sub_region'], region_dict['region'], region_dict['area_names']

    # Run model for different functions
    for model in models:

        config_base, config_model_params = deepcopy(base), deepcopy(model_params)
        config_model_params['priors'] = config['priors'][model]
        config_model_params['func'] = model
        if model == 'log_expit':  # Predictive validity only supported by Gaussian family of functions
            config_model_params['pipeline_args']['n_draws'] = 0

        if config_base['params_from_csv']:
            param_ranges = read_params_file(params_csv_path, start_date)
            config_model_params['priors']['fe_bounds'] = \
                [param_ranges['alpha'], param_ranges['beta'], param_ranges['p']]

        results = single_cycle(area_names=area_names, model_params=config_model_params,
                               which_compartments=which_compartments_enum, **config_base)

        predictions = deepcopy(results['df_prediction'])
        # If fitting to all compartments, constrain total_infected as sum of other compartments
        if set(which_compartments_enum) == set(Columns.curve_fit_compartments()):
            predictions['total_infected'] = results['df_prediction']['recovered'] + results['df_prediction'][
                'deceased'] + results['df_prediction']['hospitalised']
        results['df_final_prediction'] = predictions

        fit_plot = plot_fit(
            predictions.reset_index(), results['df_train'], results['df_val'], results['df_district'],
            train_val_period, region, sub_region, which_compartments=which_compartments,
            description='Train and test')

        forecast_plot = plot_forecast_agnostic(
            results['df_district'], predictions.reset_index(), model_name='IHME', dist=sub_region, state=region,
            which_compartments=which_compartments_enum)

        config_model_params['func'] = model
        config_model_params['ycol'] = which_compartments

        if verbose:
            print("Model params")
            print(config_model_params)

        fit_plot.savefig(os.path.join(output_folder, f'{model}.png'))
        forecast_plot.savefig(os.path.join(output_folder, f'{model}_forecast.png'))
        log_experiments(output_folder, results, config, model)


def log_experiments(output_folder, results, config, name_prefix):
    with open(os.path.join(output_folder, 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    results['df_loss'].to_csv(os.path.join(output_folder, f'{name_prefix}_loss.csv'))
    results['df_train_loss_pointwise'].to_csv(os.path.join(output_folder, f'{name_prefix}_pointwise_train_loss.csv'))
    results['df_test_loss_pointwise'].to_csv(os.path.join(output_folder, f'{name_prefix}_pointwise_test_loss.csv'))

    res_dump = deepcopy(results)
    for var in res_dump['individual_results']:
        individual_res = res_dump['individual_results'][var]
        del individual_res['optimiser']
    picklefn = os.path.join(output_folder, f'{name_prefix}.pkl')
    with open(picklefn, 'wb') as pickle_file:
        pickle.dump(res_dump, pickle_file)


def outputs(path, start=0, end=0):
    # Create output folder
    if not os.path.exists(f'{path}/consolidated'):
        os.makedirs(f'{path}/consolidated')
    # Get config
    with open(f'{path}/{start}/config.json', 'r') as infile:
        config = json.load(infile)
    config = config['base']
    models = config['models']
    compartments = config['compartments']
    test_period = config['test_size']

    param_dict = dict()
    for compartment in compartments:
        param_dict[compartment] = dict()
        for model in models:
            param_dict[compartment][model] = dict()

    for i in range(start, end + 1):
        for compartment in compartments:
            for model in models:
                picklefn = f'{path}/{i}/{model}.pkl'
                with open(picklefn, 'rb') as pickle_file:
                    model_output = pickle.load(pickle_file)
                    temp = dict(zip(["alpha", "beta", "p"], model_output['best_params'][compartment]))
                    param_dict[compartment][model][i] = temp

    for compartment in compartments:
        for model in models:
            param_dict[compartment][model] = pd.DataFrame(param_dict[compartment][model]).T
            param_dict[compartment][model].to_csv(f'{path}/consolidated/{model}_{compartment}_params.csv')

    val_loss_dict, val_loss = dict(), dict()
    for model in models:
        val_loss_dict[model] = get_ihme_loss_dict(path, f'{model}_pointwise_test_loss.csv', start=start, end=end)
        for compartment in compartments:
            val_loss = get_ihme_pointwise_loss(val_loss_dict[model], compartment=compartment, split='val',
                                               loss_fn='ape')
            create_pointwise_loss_csv(path, val_loss, test_period, model, compartment, start, end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--region_config", help="region config file name", required=True)
    parser.add_argument("-f", "--output_folder", help="output folder name", required=True)
    parser.add_argument("-n", "--num", help="number of periods to shift forward", required=False, default=1)
    parser.add_argument("-m", "--mode", help="mode", required=False, default='run')
    parser.add_argument("-s", "--start", help="start run", required=False, default=0)
    parser.add_argument("-e", "--end", help="end run", required=False, default=0)
    args = parser.parse_args()

    if args.mode == 'run':
        run_experiments(args.region_config, args.output_folder, int(args.num))
    elif args.mode == 'output':
        outputs(args.output_folder, start=int(args.start), end=int(args.end))
