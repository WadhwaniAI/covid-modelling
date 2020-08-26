"""
Script to compare variants of the IHME model fit using the erf, derf and expit functions.
"""
import argparse
import json
import os
import pickle
import sys
from copy import deepcopy
import pandas as pd

from datetime import timedelta

sys.path.append('../../')

from data.processing import get_data_from_source
from main.ihme_seir.model_runners import read_region_config, create_output_folder, ihme_runner
from main.ihme_seir.utils import get_ihme_pointwise_loss, get_ihme_loss_dict, read_config
from utils.enums import Columns
from viz.forecast import plot_errors_for_lookaheads


def run_experiments(ihme_config_path, region_config_path, data, root_folder, multiple=False, shift_forward=0):
    region_config = read_region_config(region_config_path)
    model_priors = read_region_config(region_config_path, key='priors')

    # Unpack parameters from config and set local parameters
    sub_region = region_config['sub_region']
    region = region_config['region']
    data_source = region_config['data_source']
    experiment_name = region_config['experiment_name']
    which_compartments = region_config['which_compartments']
    ihme_variants = region_config['ihme_variants']
    allowance = region_config['allowance']
    verbose = region_config['verbose']
    s1 = region_config['s1']
    s2 = region_config['s2']
    if multiple:
        shift = shift_forward
    else:
        shift = region_config['shift']

    # Enum version of compartments to fit to
    which_compartments_enum = [Columns.from_name(comp) for comp in which_compartments]

    # Set the train, val and test set sizes
    train_val_size = s1
    val_size = region_config['i1_val_size']
    test_size = s2

    # Total dataset length
    dataset_length = train_val_size + test_size

    # Set dates
    dataset_start_date = data['date'].min() + timedelta(shift)  # First date in dataset
    actual_start_date = dataset_start_date + timedelta(allowance)  # Excluding allowance at beginning

    if verbose:
        print("Training starts on:", actual_start_date)

    # Create output folder
    output_folder = create_output_folder(f'{experiment_name}/{root_folder}/')

    # Store series properties in dict
    series_properties = {
        'allowance': allowance,
        'dataset_start_date': dataset_start_date.strftime("%m-%d-%Y"),
        'shift': shift,
        'train_size': s1 - val_size,
        'val_size': val_size,
        'test_size': s2
    }

    # Run model for different functions
    for variant in ihme_variants:
        if region_config['params_from_csv']:
            params_csv_path = region_config['params_csv']
            params_df = pd.read_csv(f'config/{params_csv_path}', index_col=0, dayfirst=True)
            params_df.index = pd.to_datetime(params_df.index)
            param_ranges = dict()
            model_params = list(set([x.replace('_low', '').replace('_high', '') for x in params_df.columns]))
            for col in model_params:
                param_ranges[col] = [
                    params_df.loc[actual_start_date, :][f'{col}_low'],
                    params_df.loc[actual_start_date, :][f'{col}_high']]
            model_priors[variant]['fe_bounds'] = [param_ranges['alpha'], param_ranges['beta'], param_ranges['p']]
        variant_model_priors = model_priors[variant]
        results, config, model_params = ihme_runner(sub_region, region, actual_start_date, dataset_length,
                                                    train_val_size, val_size, test_size, ihme_config_path,
                                                    output_folder, variant=variant, model_priors=variant_model_priors,
                                                    data_source=data_source, which_compartments=which_compartments_enum)

        model_params['func'] = variant
        model_params['ycol'] = which_compartments

        if verbose:
            print("Model params")
            print(model_params)

        log_experiments(output_folder, results, config, model_params, region_config, series_properties, variant=variant)

    error_dict = dict()
    lookaheads = [i for i in range(1, test_size+1)]
    error_dict['lookahead'] = lookaheads
    error_dict['errors'] = dict()
    for compartment in which_compartments:
        for variant in ihme_variants:
            errors = pd.read_csv(f'{output_folder}ihme_{variant}_pointwise_test_loss.csv',
                                 index_col=['compartment', 'split', 'loss_function'])
            errors = errors.loc[(compartment, 'val', 'ape')].tolist()
            error_dict['errors'][variant] = errors

    plot_errors_for_lookaheads(error_dict, path=os.path.join(output_folder, 'lookahead_errors.png'))


def log_experiments(output_folder, results, model_config, model_params, region_config, series_properties,
                    variant='log_erf'):
    with open(os.path.join(output_folder, 'region_config.json'), 'w') as outfile:
        json.dump(region_config, outfile, indent=4)

    with open(os.path.join(output_folder, f'{variant}_data_params.json'), 'w') as outfile:
        json.dump(series_properties, outfile, indent=4)

    results['df_loss'].to_csv(os.path.join(output_folder, f'ihme_{variant}_loss.csv'))
    results['df_train_loss_pointwise'].to_csv(os.path.join(output_folder, f'ihme_{variant}_pointwise_train_loss.csv'))
    results['df_test_loss_pointwise'].to_csv(os.path.join(output_folder, f'ihme_{variant}_pointwise_test_loss.csv'))

    i1_config_dump = deepcopy(model_config)
    i1_config_dump['start_date'] = i1_config_dump['start_date'].strftime("%Y-%m-%d")
    with open(os.path.join(output_folder, f'ihme_{variant}_config.json'), 'w') as outfile:
        json.dump(i1_config_dump, outfile, indent=4)

    with open(os.path.join(output_folder, f'ihme_{variant}_all_params.json'), 'w') as outfile:
        json.dump(model_params, outfile, indent=4)

    res_dump = deepcopy(results)
    for var in res_dump['individual_results']:
        individual_res = res_dump['individual_results'][var]
        del individual_res['optimiser']

    picklefn = os.path.join(output_folder, f'ihme_{variant}.pkl')
    with open(picklefn, 'wb') as pickle_file:
        pickle.dump(res_dump, pickle_file)


def outputs(path, start=0, end=0):
    if not os.path.exists(f'{path}/consolidated'):
        os.makedirs(f'{path}/consolidated')
    # Get model type
    with open(f'{path}/{start}/region_config.json', 'r') as infile:
        config = json.load(infile)
    models = config['ihme_variants']
    compartments = config['which_compartments']
    val_period = config['s2']
    val_loss_dict, val_loss = dict(), dict()
    param_dict = dict()

    for compartment in compartments:
        param_dict[compartment] = dict()
        for model in models:
            param_dict[compartment][model] = dict()

    for i in range(start, end + 1):
        for compartment in compartments:
            for model in models:
                picklefn = f'{path}/{i}/ihme_{model}.pkl'
                with open(picklefn, 'rb') as pickle_file:
                    model_output = pickle.load(pickle_file)
                    temp = dict(zip(["alpha", "beta", "p"], model_output['best_params'][compartment]))
                    param_dict[compartment][model][i] = temp

    for compartment in compartments:
        for model in models:
            param_dict[compartment][model] = pd.DataFrame(param_dict[compartment][model]).T
            param_dict[compartment][model].to_csv(f'{path}/consolidated/{model}_{compartment}_params.csv')

    for model in models:
        val_loss_dict[model] = get_ihme_loss_dict(path, f'ihme_{model}_pointwise_test_loss.csv', start=start, end=end)
        for compartment in compartments:
            val_loss = get_ihme_pointwise_loss(val_loss_dict[model], compartment=compartment, split='val', loss_fn='ape')
            val_loss = pd.concat(val_loss).to_frame()
            run_num = [i for i in range(start, end + 1) for j in range(val_period)]
            lookahead = [j for i in range(start, end + 1) for j in range(val_period)]
            val_loss.insert(0, column='run', value=run_num)
            val_loss.insert(1, column='lookahead', value=lookahead)
            val_loss.to_csv(f'{path}/consolidated/{model}_{compartment}_val_loss.csv')


def runner(ihme_config_path, region_config_path, output_folder, num):
    region_config = read_region_config(region_config_path)
    sub_region = region_config['sub_region']
    region = region_config['region']
    data_source = region_config['data_source']
    shift_period = region_config['shift_period']
    shift = region_config['shift']

    # Print data summary
    data = get_data_from_source(region=region, sub_region=sub_region, data_source=data_source)

    # Output folder
    region_name = sub_region if sub_region is not None else region
    root_folder = f'{region_name}/{output_folder}'
    print(region_name, ": Run no. ", num + 1, " with shift of ", shift+shift_period * num)
    run_experiments(ihme_config_path, region_config_path, data, f'{root_folder}/{str(num)}', multiple=True,
                    shift_forward=shift+shift_period * num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ihme_config", help="ihme config file name", required=True)
    parser.add_argument("-r", "--region_config", help="region config file name", required=True)
    parser.add_argument("-f", "--output_folder", help="output folder name", required=True)
    parser.add_argument("-n", "--num", help="number of periods to shift forward", required=False, default=1)
    parser.add_argument("-m", "--mode", help="mode", required=False, default='run')
    parser.add_argument("-s", "--start", help="start run", required=False, default=0)
    parser.add_argument("-e", "--end", help="end run", required=False, default=0)
    args = parser.parse_args()

    if args.mode == 'run':
        runner(args.ihme_config, args.region_config, args.output_folder, int(args.num))
    elif args.mode == 'output':
        outputs(args.output_folder, start=int(args.start), end=int(args.end))
