import argparse
import datetime
import sys
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append('../../')

import copy
import os
import pickle

from utils.generic.config import read_config
from viz.uncertainty import plot_beta_loss


# Load all predictions_dict files_load_all_pdicts(parsed_args)
def _load_all_pdicts(parsed_args):
    predictions_dict_list = []
    for _, fname in enumerate(parsed_args.list_of_pdicts):
        predictions_pkl_filename = f'/scratche/users/sansiddh/covid-modelling/{fname}/predictions_dict.pkl'
        print(f'Reading pkl file of run {fname}')
        with open(predictions_pkl_filename, 'rb') as f:
            predictions_dict_list.append(pickle.load(f))

    return predictions_dict_list

# Create combined predictions_dict
def _create_combined_pdict(predictions_dict_list, parsed_args):
    predictions_dict_comb = {}
    for loc in predictions_dict_list[0].keys():
        predictions_dict_comb[loc] = {}
        predictions_dict_comb[loc]['plots'] = {}
        predictions_dict_comb[loc]['forecasts'] = {}
        # Copy all params
        predictions_dict_comb[loc]['run_params'] = copy.deepcopy(
            predictions_dict_list[0][loc]['run_params'])
        # Copy all dataframes
        predictions_dict_comb[loc]['df_district'] = copy.deepcopy(
            predictions_dict_list[0][loc]['df_district'])
        predictions_dict_comb[loc]['df_train'] = copy.deepcopy(
            predictions_dict_list[0][loc]['df_train'])
        predictions_dict_comb[loc]['df_val'] = copy.deepcopy(
            predictions_dict_list[0][loc]['df_val'])

        # Concatenate all predictions
        predictions_arr, losses_arr, params_arr = ([], [], [])
        for _, pdict in enumerate(predictions_dict_list):
            predictions_arr += pdict[loc]['trials']['predictions'][:parsed_args.num_trials]

            losses_arr.append(pdict[loc]['trials']['losses'][:parsed_args.num_trials])

            params_arr.append(pdict[loc]['trials']['params'][:parsed_args.num_trials])
        
        # Add them to predictions_dict_comb
        predictions_dict_comb[loc]['trials'] = {}
        predictions_dict_comb[loc]['trials']['predictions'] = predictions_arr
        predictions_dict_comb[loc]['trials']['losses'] = np.concatenate(
            tuple(losses_arr), axis=None)
        predictions_dict_comb[loc]['trials']['params'] = np.concatenate(
            tuple(params_arr), axis=None)

    return predictions_dict_comb


def _perform_beta_fitting(loc_dict, config):
    uncertainty_args = {'predictions_dict': loc_dict, 'fitting_config': config['fitting'],
                        'forecast_config': config['forecast'], 'process_trials': False,
                        **config['uncertainty']['uncertainty_params']}
    uncertainty = config['uncertainty']['method'](**uncertainty_args)

    uncertainty_forecasts = uncertainty.get_forecasts()
    for key in uncertainty_forecasts.keys():
        loc_dict['forecasts'][key] = uncertainty_forecasts[key]['df_prediction']

    loc_dict['forecasts']['ensemble_mean'] = uncertainty.ensemble_mean_forecast

    loc_dict['plots']['beta_loss'], _ = plot_beta_loss(
        uncertainty.dict_of_trials)
    loc_dict['beta'] = uncertainty.beta
    loc_dict['beta_loss'] = uncertainty.beta_loss
    loc_dict['deciles'] = uncertainty_forecasts
    
    return loc_dict

if __name__ == '__main__':

    # List of args
    parser = argparse.ArgumentParser(description="SEIR Batch Beta Fitting Script")
    parser.add_argument(
        "--config_filename", type=str, required=True, help="config filename to use while running the script"
    )
    parser.add_argument(
        "--list_of_pdicts", nargs='+', type=str, required=True,
        help="The list of folder names with the predictions_dict file which have to be merged."
    )
    parser.add_argument(
        "--username", type=str, default='sansiddh', help="Name of the user on odin/thor"
    )
    parser.add_argument(
        "--nthreads", type=int, default=40, help="Number of threads in parallel"
    )
    parser.add_argument(
        "--num_trials", type=int, default=2000, help="Number of trials for each predictions_dict to concatenate"
    )
    parsed_args = parser.parse_args()
    print(parsed_args)

    # Create folder
    output_folder = '/scratche/users/{}/covid-modelling/{}_comb'.format(
        parsed_args.username, datetime.datetime.now().strftime("%Y_%m%d_%H%M%S"))
    os.makedirs(output_folder, exist_ok=True)

    # Load and process all the args
    config = read_config(parsed_args.config_filename)
    predictions_dict_list = _load_all_pdicts(parsed_args)
    predictions_dict_comb = _create_combined_pdict(predictions_dict_list, parsed_args)

    # Do the actual fitting
    partial_perform_beta_fitting = partial(_perform_beta_fitting, config=config)

    loc_dict_arr = Parallel(n_jobs=parsed_args.nthreads)(delayed(partial_perform_beta_fitting)(
        loc_dict) for _, loc_dict in tqdm(predictions_dict_comb.items()))

    predictions_dict = dict(zip(predictions_dict_comb.keys(), loc_dict_arr))

    with open(f'{output_folder}/predictions_dict.pkl', 'wb') as f:
        pickle.dump(predictions_dict, f)
    print(f'Saved predictions_dict.')
            
