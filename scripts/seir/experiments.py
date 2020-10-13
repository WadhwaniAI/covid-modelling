"""
Script to run compartmental models.
"""
import argparse
import json
import os
import pickle
import sys
from copy import deepcopy
from datetime import timedelta, datetime

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('../../')

from data.processing.processing import get_data_from_source, get_observed_dataframes
from main.ihme_seir.utils import get_seir_pointwise_loss_dict, get_seir_pointwise_loss, read_config, read_params_file, \
    create_output_folder, create_pointwise_loss_csv, get_model, supported_models
from main.seir.fitting import get_variable_param_ranges, run_cycle
from utils.population import get_population
from utils.loss import Loss_Calculator
from utils.util import CustomEncoder
from utils.util import get_subset


def run_experiments(config_path, output_folder, num):
    output_config = dict()
    config = read_config(config_path)

    # Unpack parameters from config and set local parameters
    sub_region = config['sub_region']
    region = config['region']
    data_source = config['data_source']
    experiment_name = config['experiment_name']
    which_compartments = config['which_compartments']
    models = config['models']
    train_period = config['train_period']
    val_period = config['val_period']
    test_period = config['test_period']
    data_length = train_period + val_period + test_period
    num_evals = config['num_evals']
    shift = config['shift']
    align = config['align']
    gap = 0
    if align:
        gap = 30 - train_period  # Assuming the maximum training period is 30
    start_date = datetime.strptime(config['start_date'], '%m-%d-%Y') + timedelta(gap + shift * num)
    end_date = datetime.strptime(config['end_date'], '%m-%d-%Y')
    while (start_date + timedelta(
            train_period + val_period) > end_date+timedelta(1)) and data_length > train_period:
        val_period -= 1
        data_length -= 1
    if data_length == train_period:
        raise Exception('Insufficient data available')
    config['val_period'] = val_period
    params_csv_path = config['params_csv']
    verbose = config['verbose']
    output_config.update(config)

    if verbose:
        print("Training starts on:", start_date)

    # Get population
    N = get_population(region, sub_region)
    output_config['population'] = int(N)

    # Create output folder
    region_name = sub_region if sub_region is not None else region
    root_folder = f'{region_name}/{output_folder}/{str(num)}'
    if verbose:
        print(region_name, ": Run no. ", num + 1, " with shift of ", shift * num)
    output_folder = create_output_folder(f'{experiment_name}/{root_folder}/')

    for i, model_dict in enumerate(supported_models):
        name_prefix = model_dict['name_prefix']
        if name_prefix not in models:
            continue
        model = model_dict['model']

        # Get variable param ranges from csv or config
        if config['params_from_csv']:
            variable_param_ranges = read_params_file(params_csv_path, start_date - timedelta(gap))
        else:
            variable_param_ranges = deepcopy(config[f'params_{name_prefix}'])
        if verbose:
            print("Variable param ranges")
            print(variable_param_ranges)
        output_config[f'variable_param_ranges_{name_prefix}'] = deepcopy(variable_param_ranges)
        variable_param_ranges = get_variable_param_ranges(variable_param_ranges)

        # Get data
        data = get_data_from_source(region=region, sub_region=sub_region, data_source=data_source)
        data['daily_cases'] = data['total_infected'].diff()
        input_data = get_subset(
            data, lower=start_date, upper=start_date + timedelta(data_length - 1), col='date').reset_index(drop=True)

        # Get train and val splits
        observed_dataframes = get_observed_dataframes(input_data, val_period=val_period, test_period=test_period,
                                                      which_columns=which_compartments)

        # Run SEIR model
        predictions_dict = run_cycle(region, sub_region, observed_dataframes, model=model, data_from_tracker=True,
                                     variable_param_ranges=variable_param_ranges, train_period=train_period,
                                     which_compartments=which_compartments, num_evals=num_evals, N=N,
                                     initialisation='intermediate')

        # Log outputs
        log_experiments(output_folder, predictions_dict, output_config, which_compartments, name_prefix)


def log_experiments(output_folder, results, config, which_compartments, name_prefix):
    """Logs all results

    Args:
        output_folder (str): Output folder path
        results (dict): Model output
        config (dict): All parameters
        which_compartments (list, optional): list of compartments fitted by SEIR
        name_prefix (str): prefix for filename
    """
    with open(output_folder + 'config.json', 'w') as outfile:
        json.dump(config, outfile, indent=4)

    c1 = results['df_loss'].T[which_compartments]
    c1.to_csv(os.path.join(output_folder, f'{name_prefix}_loss.csv'))

    results['plots']['fit'].savefig(os.path.join(output_folder, f'{name_prefix}.png'))
    results['pointwise_train_loss'].to_csv(os.path.join(output_folder, f'{name_prefix}_pointwise_train_loss.csv'))
    results['pointwise_val_loss'].to_csv(os.path.join(output_folder, f'{name_prefix}_pointwise_test_loss.csv'))

    picklefn = f'{output_folder}/{name_prefix}.pkl'
    with open(picklefn, 'wb') as pickle_file:
        pickle.dump(results, pickle_file)


def outputs(path, start=0, end=0):
    # Create output folder
    if not os.path.exists(f'{path}/consolidated'):
        os.makedirs(f'{path}/consolidated')
    # Get config
    with open(f'{path}/{start}/config.json', 'r') as infile:
        config = json.load(infile)

    # Unpack parameters
    models = config['models']
    compartments = config['which_compartments']
    start_date = datetime.strptime(config['start_date'], '%m-%d-%Y')
    shift = config['shift']
    dates = pd.date_range(start=start_date, periods=(end - start + 1), freq=f'{shift}D').tolist()

    # Model params
    param_dict, param_df = dict(), dict()
    for model in models:
        param_dict[model] = dict()
    for i in range(start, end + 1):
        for model in models:
            picklefn = f'{path}/{i}/{model}.pkl'
            with open(picklefn, 'rb') as pickle_file:
                model_output = pickle.load(pickle_file)
                param_dict[model][i] = model_output['best_params']

    for model in models:
        # Save model params
        param_df[model] = pd.DataFrame(param_dict[model]).T
        param_df[model].to_csv(f'{path}/consolidated/{model}_params.csv')
        # Plot model params
        fig, ax = plt.subplots(figsize=(10, 10))
        for col in param_df[model].columns:
            ax.plot(dates, param_df[model][col], label=col)
            ax.plot(dates, param_df[model][col].rolling(window=10, center=True, min_periods=1).mean(),
                    label=f'{col} trend')
        ax.legend()
        plt.savefig(f'{path}/consolidated/{model}_params.png')
        plt.close()

    # Validation loss
    val_loss_dict, val_loss = dict(), dict()
    for model in models:
        val_loss_dict[model] = get_seir_pointwise_loss_dict(path, f'{model}_pointwise_test_loss.csv', start=start,
                                                            end=end)
        for compartment in compartments:
            val_loss = get_seir_pointwise_loss(val_loss_dict[model], compartment=compartment, loss_fn='ape')
            create_pointwise_loss_csv(path, val_loss, model, compartment, start, end)


def forecast(path, start=0, end=0):
    # Create output folder
    if not os.path.exists(f'{path}/consolidated'):
        os.makedirs(f'{path}/consolidated')
    # Get config
    with open(f'{path}/{start}/config.json', 'r') as infile:
        config = json.load(infile)

    # Unpack parameters
    models = config['models']
    compartments = config['which_compartments']
    start_date = datetime.strptime(config['start_date'], '%m-%d-%Y')
    shift = config['shift']
    train_period = config['train_period']
    test_period = config['val_period']

    data = get_data_from_source(region=config['region'], sub_region=config['sub_region'],
                                data_source=config['data_source'])
    data['daily_cases'] = data['total_infected'].diff()

    last_date = start_date + timedelta(end - start + train_period + test_period - 1)
    # Val losses
    val_loss_dict = dict()
    for model in models:
        val_loss_dict[model] = dict()
    for i in range(start, end + 1):
        for model in models:
            picklefn = f'{path}/{i}/{model}.pkl'
            with open(picklefn, 'rb') as pickle_file:
                model_output = pickle.load(pickle_file)
                predictions = model_output['optimiser'].solve(model_output['best_params'],
                                                              model_output['default_params'],
                                                              model_output['df_train'],
                                                              end_date=last_date,
                                                              model=get_model(model))

                train_data = get_subset(
                    data, lower=start_date + timedelta(i), upper=start_date + timedelta(i + train_period - 1),
                    col='date').reset_index(drop=True)
                val_data = get_subset(
                    data, lower=start_date + timedelta(i + train_period), upper=last_date,
                    col='date').reset_index(drop=True)
                lc = Loss_Calculator()

                pointwise_train_loss, pointwise_val_loss = lc.create_pointwise_loss_dataframe_region(
                    train_data, val_data, predictions, train_period, which_compartments=compartments)
                val_loss_dict[model][i] = pointwise_val_loss

        for model in models:
            for compartment in compartments:
                val_loss = get_seir_pointwise_loss(val_loss_dict[model], compartment=compartment, loss_fn='ape')
                create_pointwise_loss_csv(path, val_loss, model, compartment, start, end, outfile='test_loss_full')


def trials(path, start=0, end=0):
    # Create output folder
    if not os.path.exists(f'{path}/consolidated'):
        os.makedirs(f'{path}/consolidated')
    # Get config
    with open(f'{path}/{start}/config.json', 'r') as infile:
        config = json.load(infile)

    # Unpack parameters
    models = config['models']

    # Val losses
    val_loss_dict = dict()
    for model in models:
        val_loss_dict[model] = dict()
    for i in range(start, end + 1):
        for model in models:
            picklefn = f'{path}/{i}/{model}.pkl'
            with open(picklefn, 'rb') as pickle_file:
                model_output = pickle.load(pickle_file)
            with open(f'{path}/{i}/trials.json', 'w') as outfile:
                json.dump(model_output['trials'].__dict__['_trials'], outfile, cls=CustomEncoder)


def predictions(path, start=0, end=0):
    # Create output folder
    if not os.path.exists(f'{path}/consolidated'):
        os.makedirs(f'{path}/consolidated')
    # Get config
    with open(f'{path}/{start}/config.json', 'r') as infile:
        config = json.load(infile)

    # Unpack parameters
    models = config['models']
    compartments = config['which_compartments']
    train_period = config['train_period']

    pred_dict = dict()
    for model in models:
        pred_dict[model] = dict()
    for i in range(start, end + 1):
        for model in models:
            picklefn = f'{path}/{i}/{model}.pkl'
            with open(picklefn, 'rb') as pickle_file:
                model_output = pickle.load(pickle_file)
            pred_dict[model][i] = model_output['df_prediction'][['date', 'total_infected']]
            pred_dict[model][i].insert(0, 'run', i)
            pred_dict[model][i].insert(1, 'train_start_date', model_output['df_prediction']['date'].iloc[0])
            split = ['train']*train_period + ['forecast']*(len(model_output['df_prediction'])-train_period)
            pred_dict[model][i].insert(2, 'split', split)
    for model in models:
        pred_df = pd.concat(pred_dict[model].values()).reset_index(drop=True)
        pred_df.insert(0, 'model', model)
        pred_df.to_csv(os.path.join(path, 'consolidated', f'{model}_predictions.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--region_config", help="region config file path", required=True)
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
    elif args.mode == 'forecast':
        forecast(args.output_folder, start=int(args.start), end=int(args.end))
    elif args.mode == 'trials':
        trials(args.output_folder, start=int(args.start), end=int(args.end))
    elif args.mode == 'predictions':
        predictions(args.output_folder, start=int(args.start), end=int(args.end))
