import argparse
import sys
import numpy as np

from datetime import timedelta, datetime

sys.path.append('../../')

from utils.synthetic_data import insert_custom_dataset_into_dataframes, get_experiment_dataset, read_synth_data_config
from utils.loss import Loss_Calculator

from data.dataloader import Covid19IndiaLoader
from data.processing import get_data

from main.ihme.synthetic_data_generator import *
from main.seir.fitting import get_regional_data

from viz.synthetic_data import plot_all_experiments, plot_against_baseline


def run_experiments(ihme_config_path, data_config_path, dataframes, data, multiple=False, shift_forward=0):
    data_config = read_synth_data_config(data_config_path)

    # Unpack parameters from config and set local parameters
    district = data_config['district']
    state = data_config['state']
    disable_tracker = data_config['disable_tracker']
    allowance = data_config['allowance']
    s1 = data_config['s1']
    s2 = data_config['s2']
    s3 = data_config['s3']
    if multiple:
        shift = shift_forward
    else:
        shift = data_config['shift']

    i1_train_val_size = s1
    i1_val_size = data_config['i1_val_size']
    i1_test_size = s2

    c1_train_period = s1
    c1_val_period = s2
    c2_train_period = data_config['c2_train_period']
    c2_val_period = s3
    c3_train_period = data_config['c3_train_period']
    c3_val_period = s2 + s3

    num_exp = 3
    num_evals = 1500

    i1_dataset_length = i1_train_val_size + i1_test_size
    c1_dataset_length = shift + allowance + c1_train_period + c1_val_period

    smooth_jump = True if district == "Mumbai" else False
    which_compartments = ['hospitalised', 'total_infected', 'deceased', 'recovered']

    # Set dates
    dataset_start_date = data['date'].min() + timedelta(shift)  # First date in dataset
    actual_start_date = dataset_start_date + timedelta(allowance)  # Excluding allowance at beginning

    # Create output folder
    now = datetime.now()
    date_now = now.strftime("%Y%m%d")
    time_now = now.strftime("%H%M%S")
    folder = f'{district}/{str(date_now)}/{str(time_now)}'
    output_folder = create_output_folder(f'synth/{folder}/')

    # Store series properties in dict
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
        'c2_val_period': c2_val_period,
        'c3_train_period': c3_train_period,
        'c3_val_period': c3_val_period
    }

    # Generate synthetic data using IHME model
    i1_output, i1_config, i1_model_params = ihme_data_generator(district, disable_tracker,
                                                                actual_start_date, i1_dataset_length,
                                                                i1_train_val_size, i1_val_size, i1_test_size,
                                                                ihme_config_path, output_folder)

    # IHME uncertainty
    draws = i1_output['draws']
    average_uncertainty_s2 = dict()
    for compartment in which_compartments:
        draws_compartment = draws[compartment]['draws']
        draws_compartment_s2 = draws_compartment[:, s1:s1 + s2]
        average_uncertainty_s2[compartment] = np.mean(draws_compartment_s2[1] - draws_compartment_s2[0])
    uncertainty = pd.DataFrame.from_dict(average_uncertainty_s2, orient='index', columns=['average s2 uncertainty'])
    i1_output['df_loss'] = pd.concat([i1_output['df_loss'], uncertainty], axis=1)

    # Get SEIR input dataframes
    input_df = get_regional_data(dataframes, state, district, (not disable_tracker), None, None, granular_data=False,
                                 smooth_jump=smooth_jump, smoothing_length=33, smoothing_method='weighted', t_recov=14,
                                 return_extra=False, which_compartments=which_compartments)

    # Generate synthetic data using SEIR model
    input_df_c1 = deepcopy(input_df)
    df_district, df_raw = input_df_c1
    df_district = df_district.head(c1_dataset_length)
    input_df_c1 = (df_district, df_raw)
    c1_output = seir_runner(district, state, input_df_c1, (not disable_tracker),
                            c1_train_period, c1_val_period, which_compartments, num_evals=num_evals)

    original_data = deepcopy(i1_output['df_district_nora'])

    # Set experiment data configs
    exp_config = [deepcopy(data_config) for _ in range(3)]
    exp_config[0]['use_synthetic'] = False
    exp_config[0]['generated_data'] = None
    exp_config[1]['generated_data'] = i1_output['df_final_prediction']
    exp_config[2]['generated_data'] = c1_output['m1']['df_prediction']

    # Get custom datasets for experiments
    df, train, test, dataset_prop = dict(), dict(), dict(), dict()
    for exp in range(num_exp):
        df[exp], train[exp], test[exp], dataset_prop[exp] = get_experiment_dataset(
            district, state, original_data, generated_data=exp_config[exp]['generated_data'],
            use_actual=exp_config[exp]['use_actual'], use_synthetic=exp_config[exp]['use_synthetic'],
            start_date=dataset_start_date, allowance=allowance, s1=s1, s2=s2, s3=s3)

    # Insert custom data into SEIR input dataframes
    input_dfs = dict()
    for exp in range(num_exp):
        input_dfs[exp] = insert_custom_dataset_into_dataframes(input_df, df[exp],
                                                               start_date=dataset_start_date,
                                                               compartments=which_compartments)

    # Get SEIR predictions on custom datasets
    predictions_dicts = dict()
    for exp in range(num_exp):
        predictions_dicts[exp] = seir_runner(district, state, input_dfs[exp], (not disable_tracker),
                                             c2_train_period, c2_val_period, which_compartments, num_evals=num_evals)

    # Get baseline c3 predictions for s2+s3 when trained on s1
    df_baseline, train_baseline, test_baseline, dataset_prop_baseline = get_experiment_dataset(
        district, state, original_data, generated_data=None, use_actual=True, use_synthetic=False,
        start_date=dataset_start_date, allowance=allowance, s1=s1, s2=0, s3=s2+s3)
    input_df_baseline = insert_custom_dataset_into_dataframes(input_df, df_baseline, start_date=dataset_start_date,
                                                              compartments=which_compartments)
    predictions_dict_baseline = seir_runner(district, state, input_df_baseline, (not disable_tracker),
                                            c3_train_period, c3_val_period, which_compartments, num_evals=num_evals)

    # Find loss on s3 for baseline c3 model
    lc = Loss_Calculator()
    df_c2_s3_loss = lc.create_loss_dataframe_region(train_baseline[-c3_train_period:], test_baseline[-s3:],
                                                    predictions_dict_baseline['m1']['df_prediction'],
                                                    c3_train_period, which_compartments=which_compartments)
    predictions_dict_baseline['m1']['df_loss_s3'] = df_c2_s3_loss

    # Plotting all experiments
    plot_all_experiments(input_df[0], predictions_dicts, district, actual_start_date,
                         allowance, s1, s2, s3, shift, c2_train_period, output_folder)

    # Plot performance against baseline
    plot_against_baseline(input_df[0], predictions_dicts, predictions_dict_baseline, district,
                          actual_start_date, allowance, s1, s2, s3, shift, c2_train_period, c3_train_period,
                          output_folder)

    # Log results
    log_experiment_local(output_folder, i1_config, i1_model_params, i1_output,
                         c1_output, df, predictions_dicts, which_compartments,
                         dataset_prop, series_properties, baseline_predictions_dict=predictions_dict_baseline)


def run_experiments_over_time(ihme_config_path, data_config_path, num, shift):
    data_config = read_synth_data_config(data_config_path)
    district = data_config['district']
    state = data_config['state']
    disable_tracker = data_config['disable_tracker']

    # Check data if tracker data is used
    loader = Covid19IndiaLoader()
    dataframes = None
    if not disable_tracker:
        dataframes = loader.get_covid19india_api_data()

    # Print data summary
    data = get_data(dataframes, state, district, disable_tracker=disable_tracker)
    print("Data summary:")
    print("Start date:", data.iloc[0]['date'])
    print("End date:", data.iloc[-1]['date'])
    print("Number of rows:", data.shape[0])

    if num == 1:
        run_experiments(ihme_config_path, data_config_path, dataframes, data, multiple=False, shift_forward=0)
    else:
        for i in range(num):
            print("Run no. ", i+1)
            run_experiments(ihme_config_path, data_config_path, dataframes, data, multiple=True, shift_forward=shift*i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ihme_config", help="ihme config file name", required=True)
    parser.add_argument("-d", "--data_config", help="data config file name", required=True)
    parser.add_argument("-n", "--num", help="number of times experiments are run", required=False, default=1)
    parser.add_argument("-s", "--shift", help="number of days to shift forward", required=False, default=5)
    args = parser.parse_args()

    run_experiments_over_time(args.ihme_config, args.data_config, int(args.num), int(args.shift))
