import argparse
import sys

from datetime import timedelta, datetime

sys.path.append('../../')

from utils.synthetic_data import insert_custom_dataset_into_dataframes, get_experiment_dataset, read_synth_data_config

from data.dataloader import Covid19IndiaLoader
from data.processing import get_data

from main.ihme.fitting import create_output_folder
from main.ihme.synthetic_data_generator import *
from main.seir.fitting import get_regional_data

from viz.synthetic_data import plot_all_experiments


def run_experiments(ihme_config, data_config):
    data_config = read_synth_data_config(data_config)

    # Unpack parameters from config and set local parameters
    district = data_config['district']
    state = data_config['state']
    disable_tracker = data_config['disable_tracker']
    allowance = data_config['allowance']
    s1 = data_config['s1']
    s2 = data_config['s2']
    s3 = data_config['s3']
    shift = data_config['shift']
    i1_val_size = data_config['i1_val_size']
    c1_train_period = s1
    c1_val_period = s2
    c2_train_period = data_config['c2_train_period']
    c2_val_period = s3
    num_exp = 3
    period = s1 + s2
    i1_test_size = s2
    c1_dataset_length = shift + allowance + c1_train_period + c1_val_period
    smooth_jump = True if district == "Mumbai" else False
    which_compartments = ['hospitalised', 'total_infected', 'deceased', 'recovered']

    # Check data if tracker data is used
    loader = Covid19IndiaLoader()
    dataframes = loader.get_covid19india_api_data()

    # Print data summary
    data = get_data(dataframes, state, district, disable_tracker=disable_tracker)
    print("Data summary:")
    print("Start:")
    print(data.iloc[0])
    print("End:")
    print(data.iloc[-1])
    print("Number of rows:", data.shape[0])

    # Set dates
    dataset_start_date = data['date'].min() + timedelta(shift)  # First date in dataset
    actual_start_date = dataset_start_date + timedelta(allowance)  # Excluding allowance at beginning

    # Create output folder
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = f'{district}/{str(now)}'
    output_folder = create_output_folder(f'synth/{folder}/')

    series_properties = {
        'allowance': allowance,
        's1': s1,
        's2': s2,
        's3': s3,
        'dataset_start_date': dataset_start_date.strftime("%m-%d-%Y"),
        'shift': shift,
        'i1_train_size': s1-i1_val_size,
        'i1_val_size': i1_val_size,
        'i1_test_size': s2,
        'c1_train_period': c1_train_period,
        'c1_val_period': c1_val_period,
        'c2_train_period': c2_train_period,
        'c2_val_period': c2_val_period
    }

    # Generate synthetic data using IHME model
    ihme_synthetic_data, ihme_config, ihme_model_params = ihme_data_generator(district, ihme_config, output_folder,
                                                                              disable_tracker, actual_start_date,
                                                                              period, s1, i1_val_size, i1_test_size)

    # Generate synthetic data using SEIR model
    seir_synthetic_data = seir_data_generator(dataframes, district, state, disable_tracker, smooth_jump,
                                              c1_train_period, c1_val_period, c1_dataset_length)

    original_data = deepcopy(ihme_synthetic_data['df_district_nora'])

    # Set experiment data configs
    exp_config = [deepcopy(data_config) for _ in range(3)]
    exp_config[0]['use_synthetic'] = False
    exp_config[0]['generated_data'] = ihme_synthetic_data['df_final_prediction']
    exp_config[1]['generated_data'] = ihme_synthetic_data['df_final_prediction']
    exp_config[2]['generated_data'] = seir_synthetic_data['m1']['df_prediction']

    # Get custom datasets for experiments
    df, train, test, dataset_prop = dict(), dict(), dict(), dict()
    for exp in range(num_exp):
        df[exp], train[exp], test[exp], dataset_prop[exp] = get_experiment_dataset(original_data,
                                                                                   exp_config[exp]['generated_data'],
                                                                                   state, district)

    # Get SEIR input dataframes
    input_df = get_regional_data(dataframes, state, district, (not disable_tracker), None, None, granular_data=False,
                                 smooth_jump=smooth_jump, smoothing_length=33, smoothing_method='weighted', t_recov=14,
                                 return_extra=False)

    # Insert custom data into SEIR input dataframes
    input_dfs = dict()
    for exp in range(num_exp):
        input_dfs[exp] = insert_custom_dataset_into_dataframes(input_df, df[exp],
                                                               start_date=dataset_start_date,
                                                               compartments=which_compartments)

    # Get SEIR predictions on custom datasets
    predictions_dicts = dict()
    for exp in range(num_exp):
        predictions_dicts[exp] = seir_with_synthetic_data(state, district, input_dfs[exp], (not disable_tracker),
                                                          c2_train_period, c2_val_period, which_compartments)

    # Plotting
    plot_all_experiments(input_df[0], predictions_dicts, district, actual_start_date,
                         allowance, s1, s2, s3, shift, c2_train_period, output_folder)

    # Log results
    log_experiment_local(output_folder, ihme_config, ihme_model_params, ihme_synthetic_data,
                         seir_synthetic_data, df, predictions_dicts, which_compartments,
                         dataset_prop, series_properties)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ihme_config", help="ihme config file name", required=True)
    parser.add_argument("-d", "--data_config", help="data config file name", required=True)
    args = parser.parse_args()

    run_experiments(args.ihme_config, args.data_config)
