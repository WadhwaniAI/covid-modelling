import os
import pickle
import argparse
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.generic.config import read_config, make_date_key_str
from viz import plot_ptiles


def _perform_beta_fitting(loc_dict, config):
    uncertainty_args = {'predictions_dict': loc_dict, 'fitting_config': config['fitting'],
                        'forecast_config': config['forecast'], 'process_trials': False,
                        **config['uncertainty']['uncertainty_params']}
    uncertainty = config['uncertainty']['method'](**uncertainty_args)

    uncertainty_forecasts = uncertainty.get_forecasts()
    for key in uncertainty_forecasts.keys():
        loc_dict['m2']['forecasts'][key] = uncertainty_forecasts[key]['df_prediction']

    loc_dict['m2']['forecasts']['ensemble_mean'] = uncertainty.ensemble_mean_forecast

    ptiles_plots = plot_ptiles(loc_dict,
                               which_compartments=config['forecast']['plot_ptiles_for_columns'])

    loc_dict['m2']['plots']['forecasts_ptiles'] = {}
    for column in config['forecast']['plot_ptiles_for_columns']:
        loc_dict['m2']['plots']['forecasts_ptiles'][column.name] = ptiles_plots[column]
    
    return loc_dict


parser = argparse.ArgumentParser(description="SEIR Batch Beta Fitting Script")
parser.add_argument(
    "--config_filename", type=str, required=True, help="config filename to use while running the script"
)
parser.add_argument(
    "--pdict_name", type=str, required=True, help="The folder name which has the predictions_dict file"
)
parser.add_argument(
    "--username", type=str, default='sansiddh', help="Name of the user on odin/thor"
)
parser.add_argument(
    "--nthreads", type=int, default=40, help="Number of threads in parallel"
)
parsed_args = parser.parse_args()
print(parsed_args)
config = read_config(parsed_args.config_filename)

wandb_config = read_config(parsed_args.config_filename, preprocess=False)
wandb_config = make_date_key_str(wandb_config)

predictions_pkl_filename = f'/scratche/users/{parsed_args.username}/covid-modelling/' + \
    f'{parsed_args.pdict_name}/predictions_dict.pkl'
with open(predictions_pkl_filename, 'rb') as f:
    predictions_dict = pickle.load(f)

partial_run_beta_fitting = partial(_perform_beta_fitting, config=config)

loc_dict_arr = Parallel(n_jobs=parsed_args.nthreads)(delayed(partial_run_beta_fitting)(
    loc_dict) for _, loc_dict in tqdm(predictions_dict.items()))

predictions_dict = dict(zip(predictions_dict.keys(), loc_dict_arr))

output_folder = f'/scratche/users/{parsed_args.username}/covid-modelling/' + \
    f'{parsed_args.pdict_name}_beta'
os.makedirs(output_folder, exist_ok=True)
with open(f'{output_folder}/predictions_dict.pkl', 'wb') as f:
    pickle.dump(predictions_dict, f)
