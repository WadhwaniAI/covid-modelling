from copy import deepcopy
from datetime import timedelta

from utils.synthetic_data import insert_custom_dataset_into_dataframes, get_experiment_dataset
from utils.loss import Loss_Calculator
from utils.enums import Columns

from main.ihme_seir.synthetic_data_generator import ihme_data_generator, seir_runner, log_experiment_local, \
    create_output_folder, get_variable_param_ranges_dict, read_region_config, supported_models
from main.seir.fitting import get_regional_data, get_variable_param_ranges

from viz.synthetic_data import plot_all_experiments, plot_against_baseline


def run_experiments(ihme_config_path, region_config_path, data, root_folder, multiple=False, shift_forward=0):
    """Runs a set of experiments for a region

    Args:
        ihme_config_path (str): path to ihme config
        region_config_path (str): path to experiment config for region
        data (pd.DataFrame): data for region
        root_folder (str): path to output folder
        multiple (bool): if True, use shift_forward argument, otherwise use shift from config
        shift_forward (int): day number on which experiments start (with day 0 as first available date of data)

    """

    region_config = read_region_config(region_config_path)

    # Unpack parameters from config and set local parameters
    district = region_config['district']
    state = region_config['state']
    experiment_name = region_config['experiment_name']
    replace_compartments = region_config['replace_compartments']
    which_compartments = region_config['which_compartments']
    compartmental_models = region_config['compartmental_models']
    disable_tracker = region_config['disable_tracker']
    allowance = region_config['allowance']
    s1 = region_config['s1']
    s2 = region_config['s2']
    s3 = region_config['s3']
    if multiple:
        shift = shift_forward
    else:
        shift = region_config['shift']

    i1_train_val_size = s1
    i1_val_size = region_config['i1_val_size']
    i1_test_size = s2

    c1_train_period = s1
    c1_val_period = s2
    c2_train_period = region_config['c2_train_period']
    c2_val_period = s3
    c3_train_period = region_config['c3_train_period']
    c3_val_period = s2 + s3

    num_exp = 3
    num_evals = region_config['num_evals']

    i1_dataset_length = i1_train_val_size + i1_test_size
    c1_dataset_length = shift + allowance + c1_train_period + c1_val_period

    smooth_jump = True if district == "Mumbai" else False

    replace_compartments_enum = [Columns.from_name(comp) for comp in replace_compartments]

    # Set dates
    dataset_start_date = data['date'].min() + timedelta(shift)  # First date in dataset
    actual_start_date = dataset_start_date + timedelta(allowance)  # Excluding allowance at beginning

    # Create output folder
    output_folder = create_output_folder(f'{experiment_name}/{root_folder}/')

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
    print("IHME I1 model")
    i1_output, i1_config, i1_model_params = ihme_data_generator(district, state, disable_tracker,
                                                                actual_start_date, i1_dataset_length,
                                                                i1_train_val_size, i1_val_size, i1_test_size,
                                                                ihme_config_path, output_folder,
                                                                which_compartments=replace_compartments_enum)

    # Get SEIR input dataframes
    input_df = get_regional_data(state, district, (not disable_tracker), None, None, granular_data=False,
                                 smooth_jump=smooth_jump, t_recov=14,
                                 return_extra=False, which_compartments=which_compartments)

    for i, model_dict in enumerate(supported_models):
        name_prefix = model_dict['name_prefix']
        if name_prefix not in compartmental_models:
            continue
        model = model_dict['model']
        name = model_dict['name']
        variable_param_ranges = get_variable_param_ranges_dict(district, state,
                                                               model_type=name_prefix, train_period=c2_train_period)
        variable_param_ranges_copy = deepcopy(variable_param_ranges)
        variable_param_ranges = get_variable_param_ranges(variable_param_ranges)

        # Generate synthetic data using SEIR model
        input_df_c1 = deepcopy(input_df)
        input_df_c1 = input_df_c1.head(c1_dataset_length)

        print(name, " C1 model")
        c1_output = seir_runner(district, state, input_df_c1, (not disable_tracker),
                                c1_train_period, c1_val_period, which_compartments,
                                model=model, variable_param_ranges=variable_param_ranges,
                                num_evals=num_evals)

        original_data = deepcopy(i1_output['df_district_nora'])

        # Set experiment data configs
        exp_config = [deepcopy(region_config) for _ in range(3)]
        exp_config[0]['use_synthetic'] = False
        exp_config[0]['generated_data'] = None
        exp_config[1]['generated_data'] = i1_output['df_final_prediction']
        exp_config[2]['generated_data'] = c1_output['df_prediction']

        # Get custom datasets for experiments
        df, train, test, dataset_prop = dict(), dict(), dict(), dict()
        for exp in range(num_exp):
            df[exp], train[exp], test[exp], dataset_prop[exp] = get_experiment_dataset(
                district, state, original_data, generated_data=exp_config[exp]['generated_data'],
                use_actual=exp_config[exp]['use_actual'], use_synthetic=exp_config[exp]['use_synthetic'],
                start_date=dataset_start_date, allowance=allowance, s1=s1, s2=s2, s3=s3,
                compartments=replace_compartments_enum)

        # Insert custom data into SEIR input dataframes
        input_dfs = dict()
        for exp in range(num_exp):
            input_dfs[exp] = insert_custom_dataset_into_dataframes(input_df, df[exp], start_date=dataset_start_date,
                                                                   compartments=replace_compartments_enum)

        # Get SEIR predictions on custom datasets
        print(name, " C2 model")
        predictions_dicts = dict()
        for exp in range(num_exp):
            predictions_dicts[exp] = seir_runner(district, state, input_dfs[exp], (not disable_tracker),
                                                 c2_train_period, c2_val_period, which_compartments,
                                                 model=model, variable_param_ranges=variable_param_ranges,
                                                 num_evals=num_evals)

        # Get baseline c3 predictions for s2+s3 when trained on s1
        print(name, " C3 model")
        df_baseline, train_baseline, test_baseline, dataset_prop_baseline = get_experiment_dataset(
            district, state, original_data, generated_data=None, use_actual=True, use_synthetic=False,
            start_date=dataset_start_date, allowance=allowance, s1=s1, s2=0, s3=s2+s3,
            compartments=replace_compartments_enum)
        input_df_baseline = insert_custom_dataset_into_dataframes(input_df, df_baseline, start_date=dataset_start_date,
                                                                  compartments=replace_compartments_enum)
        predictions_dict_baseline = seir_runner(district, state, input_df_baseline, (not disable_tracker),
                                                c3_train_period, c3_val_period, which_compartments,
                                                model=model, variable_param_ranges=variable_param_ranges,
                                                num_evals=num_evals)

        # Find loss on s3 for baseline c3 model
        lc = Loss_Calculator()
        df_c3_s3_loss = lc.create_loss_dataframe_region(train_baseline[-c3_train_period:], test_baseline[-s3:],
                                                        predictions_dict_baseline['df_prediction'],
                                                        c3_train_period, which_compartments=replace_compartments)
        predictions_dict_baseline['df_loss_s3'] = df_c3_s3_loss

        print("Creating plots...")
        # Plotting all experiments
        plot_all_experiments(input_df, predictions_dicts, district, actual_start_date,
                             allowance, s1, s2, s3, shift, c2_train_period, output_folder, name_prefix=name_prefix,
                             which_compartments=replace_compartments_enum)

        # Plot performance against baseline
        plot_against_baseline(input_df, predictions_dicts, predictions_dict_baseline, district,
                              actual_start_date, allowance, s1, s2, s3, shift, c2_train_period, c3_train_period,
                              output_folder, name_prefix=name_prefix, which_compartments=replace_compartments_enum)

        # Log results
        print("Saving results...")
        log_experiment_local(output_folder, region_config, i1_config, i1_model_params, i1_output,
                             c1_output, df, predictions_dicts, which_compartments, replace_compartments,
                             dataset_prop, series_properties, baseline_predictions_dict=predictions_dict_baseline,
                             name_prefix=name_prefix, variable_param_ranges=variable_param_ranges_copy)
