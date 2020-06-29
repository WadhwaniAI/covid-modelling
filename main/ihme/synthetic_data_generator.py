import os
import json
import pandas as pd

from copy import deepcopy

from utils.enums import Columns
from utils.util import read_config
from utils.data import cities

from main.ihme.fitting import single_cycle
from main.seir.fitting import get_regional_data, data_setup, run_cycle

from models.seir.seir_testing import SEIR_Testing

from viz.forecast import plot_forecast_agnostic
from viz.fit import plot_fit


def ihme_data_generator(district, config_path, output_folder,
                        disable_tracker, actual_start_date, period,
                        train_size, val_size, test_size):
    district, state, area_names = cities[district.lower()]
    config, model_params = read_config(config_path)

    config['start_date'] = actual_start_date
    config['period'] = period
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
        train_size, state, district, which_compartments=[c.name for c in Columns.curve_fit_compartments()],
        description='Train and test',
        savepath=os.path.join(output_folder, 'ihme-i1-fit.png'))

    plot_forecast_agnostic(ihme_df_true, makesum.reset_index(), model_name='IHME',
                           dist=district, state=state, filename=os.path.join(output_folder, 'ihme-i1-forecast.png'))

    return ihme_res, config, model_params


def seir_data_generator(dataframes, district, state, disable_tracker, smooth_jump,
                        train_period, val_period, dataset_length):
    model = SEIR_Testing
    data_from_tracker = (not disable_tracker)
    which_compartments = ['hospitalised', 'total_infected', 'deceased', 'recovered']
    smooth_jump = smooth_jump
    input_df = get_regional_data(dataframes, state, district, (not disable_tracker), None, None, granular_data=False,
                                 smooth_jump=smooth_jump, smoothing_length=33, smoothing_method='weighted', t_recov=14,
                                 return_extra=False)

    df_district, df_raw = input_df
    df_district = df_district.head(dataset_length)
    predictions_dict = dict()
    observed_dataframes = data_setup(df_district, df_raw, val_period)

    predictions_dict['m1'] = run_cycle(
        state, district, observed_dataframes,
        model=model, variable_param_ranges=None,
        data_from_tracker=data_from_tracker, train_period=train_period,
        which_compartments=which_compartments, N=1e7,
        num_evals=1500, initialisation='intermediate'
    )

    return predictions_dict


def seir_with_synthetic_data(state, district, input_df, data_from_tracker,
                             train_period, val_period, which_compartments):
    predictions_dict = dict()
    observed_dataframes = data_setup(input_df[0], input_df[1], val_period)
    predictions_dict['m1'] = run_cycle(
        state, district, observed_dataframes,
        model=SEIR_Testing, variable_param_ranges=None,
        data_from_tracker=data_from_tracker, train_period=train_period,
        which_compartments=which_compartments, N=1e7,
        num_evals=1500, initialisation='intermediate'
    )
    return predictions_dict


def log_experiment_local(output_folder, ihme_config, ihme_model_params, ihme_synthetic_data,
                         seir_synthetic_data, datasets, predictions_dicts, which_compartments,
                         dataset_properties, series_properties):
    params_dict = {
        'compartments_replaced': which_compartments,
        'dataset_properties': {
            'exp' + str(i): dataset_properties[i] for i in range(len(dataset_properties))
        },
        'series_properties': series_properties
    }

    for i in datasets:
        filename = 'dataset_' + str(i+1) + '.csv'
        datasets[i].to_csv(output_folder+filename)

    c1 = seir_synthetic_data['m1']['df_loss'].T[which_compartments]
    c1.to_csv(output_folder + "seir_loss.csv")

    seir_synthetic_data['m1']['ax'].figure.savefig(output_folder + 'seir_c1.png')

    i1 = ihme_synthetic_data['df_loss'].T[which_compartments]
    i1.to_csv(output_folder + "ihme_loss.csv")

    loss_dfs = []
    for i in predictions_dicts:
        loss_df = predictions_dicts[i]['m1']['df_loss'].T[which_compartments]
        loss_df['exp'] = i+1
        loss_dfs.append(loss_df)
        predictions_dicts[i]['m1']['ax'].figure.savefig(output_folder + 'seir_exp'+str(i+1)+'.png')

    loss = pd.concat(loss_dfs, axis=0)
    loss.index.name = 'index'
    loss.sort_values(by='index', inplace=True)
    loss.to_csv(output_folder + "/exp_" + "_".join(which_compartments) + ".csv")

    with open(output_folder + 'params.json', 'w') as outfile:
        json.dump(params_dict, outfile, indent=4)

    ihme_config['start_date'] = ihme_config['start_date'].strftime("%Y-%m-%d")
    with open(output_folder + 'ihme_config.json', 'w') as outfile:
        json.dump(ihme_config, outfile, indent=4)

    with open(output_folder + 'ihme_model_params.json', 'w') as outfile:
        json.dump(repr(ihme_model_params), outfile, indent=4)
