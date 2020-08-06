"""
Script to compare variants of the compartmental model: SIRD and SEIHRD.
"""
import argparse
import json
import os
import pickle
import sys
from copy import deepcopy
from datetime import timedelta

sys.path.append('../../')

from data.processing import get_data_from_source
from main.ihme_seir.synthetic_data_generator import read_region_config, create_output_folder, supported_models, \
    get_variable_param_ranges_dict, seir_runner
from main.seir.fitting import get_regional_data, get_variable_param_ranges
from utils.population import get_population


def run_experiments(region_config_path, data, root_folder, multiple=False, shift_forward=0):
    region_config = read_region_config(region_config_path)

    # Unpack parameters from config and set local parameters
    sub_region = region_config['sub_region']
    region = region_config['region']
    experiment_name = region_config['experiment_name']
    which_compartments = region_config['which_compartments']
    compartmental_models = region_config['compartmental_models']
    disable_tracker = region_config['disable_tracker']
    allowance = region_config['allowance']
    s1 = region_config['s1']
    s2 = region_config['s2']
    if multiple:
        shift = shift_forward
    else:
        shift = region_config['shift']
    N = get_population(region, sub_region)

    c1_train_period = s1
    c1_val_period = s2

    num_evals = region_config['num_evals']

    c1_dataset_length = shift + allowance + c1_train_period + c1_val_period

    smooth_jump = True if sub_region == "Mumbai" else False

    # Set dates
    dataset_start_date = data['date'].min() + timedelta(shift)  # First date in dataset

    # Create output folder
    output_folder = create_output_folder(f'{experiment_name}/{root_folder}/')

    # Store series properties in dict
    series_properties = {
        'allowance': allowance,
        'dataset_start_date': dataset_start_date.strftime("%m-%d-%Y"),
        'shift': shift,
        'train_period': c1_train_period,
        'val_period': c1_val_period,
    }

    # Get SEIR input dataframes
    if smooth_jump:
        input_df = get_regional_data(region, sub_region, (not disable_tracker), None, None, granular_data=False,
                                     smooth_jump=smooth_jump, t_recov=14,
                                     return_extra=False, which_compartments=which_compartments)
    else:
        input_df = deepcopy(data)
    input_df['daily_cases'] = input_df['total_infected'].diff()

    for i, model_dict in enumerate(supported_models):
        name_prefix = model_dict['name_prefix']
        if name_prefix not in compartmental_models:
            continue
        model = model_dict['model']
        variable_param_ranges = get_variable_param_ranges_dict(sub_region, region,
                                                               model_type=name_prefix, train_period=None)
        variable_param_ranges_copy = deepcopy(variable_param_ranges)
        variable_param_ranges = get_variable_param_ranges(variable_param_ranges)

        input_df_c1 = deepcopy(input_df)
        input_df_c1 = input_df_c1.head(c1_dataset_length)

        print(input_df_c1)

        c1_output = seir_runner(sub_region, region, input_df_c1, (not disable_tracker), c1_train_period, c1_val_period,
                                which_compartments, model=model, variable_param_ranges=variable_param_ranges, N=N,
                                num_evals=num_evals)

        log_experiments(output_folder, region_config, c1_output, which_compartments, series_properties, name_prefix,
                        variable_param_ranges=variable_param_ranges_copy)


def log_experiments(output_folder, region_config, c1_output, which_compartments, series_properties, name_prefix="",
                    variable_param_ranges=None):
    """Logs all results

    Args:
        output_folder (str): Output folder path
        region_config (dict): Config for experiments
        c1_output (dict): Results dict of SEIR C1 model
        which_compartments (list, optional): list of compartments fitted by SEIR
        series_properties (dict): Properties of series used in experiments
        name_prefix (str): prefix for filename
        variable_param_ranges(dict): parameter search spaces
    """
    with open(os.path.join(output_folder, f'{name_prefix}_data_params.json'), 'w') as outfile:
        json.dump(series_properties, outfile, indent=4)

    with open(output_folder + 'region_config.json', 'w') as outfile:
        json.dump(region_config, outfile, indent=4)

    c1 = c1_output['df_loss'].T[which_compartments]
    c1.to_csv(output_folder + name_prefix + "_loss.csv")

    c1_output['plots']['fit'].savefig(output_folder + name_prefix + '.png')
    c1_output['pointwise_train_loss'].to_csv(output_folder + name_prefix + "_pointwise_train_loss.csv")
    c1_output['pointwise_val_loss'].to_csv(output_folder + name_prefix + "_pointwise_val_loss.csv")

    with open(f'{output_folder}{name_prefix}_best_params.json', 'w') as outfile:
        json.dump(c1_output['best_params'], outfile, indent=4)
    if variable_param_ranges is not None:
        with open(f'{output_folder}{name_prefix}_variable_param_ranges.json', 'w') as outfile:
            json.dump(variable_param_ranges, outfile, indent=4)

    picklefn = f'{output_folder}/{name_prefix}.pkl'
    with open(picklefn, 'wb') as pickle_file:
        pickle.dump(c1_output, pickle_file)


def runner(region_config_path, output_folder, num):
    region_config = read_region_config(region_config_path)
    sub_region = region_config['sub_region']
    region = region_config['region']
    data_source = region_config['data_source']
    disable_tracker = region_config['disable_tracker']
    shift_period = region_config['shift_period']

    # Print data summary
    data = get_data_from_source(region=region, sub_region=sub_region, data_source=data_source,
                                disable_tracker=disable_tracker)
    print(data)

    # Output folder
    region_name = sub_region if sub_region is not None else region
    root_folder = f'{region_name}/{output_folder}'
    print(region_name, ": Run no. ", num + 1, " with shift of ", shift_period * num)
    run_experiments(region_config_path, data, f'{root_folder}/{str(num)}', multiple=False,
                    shift_forward=shift_period * num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--region_config", help="region config file name", required=True)
    parser.add_argument("-f", "--output_folder", help="output folder name", required=True)
    parser.add_argument("-n", "--num", help="number of periods to shift forward", required=False, default=1)
    args = parser.parse_args()

    runner(args.region_config, args.output_folder, int(args.num))
