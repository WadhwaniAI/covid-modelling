import argparse
import copy
import os
import pickle
from functools import partial
from glob import glob
from os.path import basename, dirname, join

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.generic.config import read_config
from viz.uncertainty import plot_beta_loss
from utils.fitting.util import create_output


# Load all predictions_dict files_load_all_pdicts(parsed_args)
def _load_all_pdicts(base_folder, list_of_pdicts):
    predictions_dict_list = []
    for _, fname in enumerate(list_of_pdicts):
        output_foldername = join(base_folder, f'{fname}')
        print(f'Reading pkl file of run {fname}')
        predictions_dict = {}
        with open(join(output_foldername, 'trials.pkl'), 'rb') as f:
            predictions_dict['trials'] = pickle.load(f)

        with open(join(output_foldername, 'run_params.pkl'), 'rb') as f:
            predictions_dict['run_params'] = pickle.load(f)
            
        predictions_dict['df_train'] = pd.read_csv(join(output_foldername, 'df_train.csv'))
        predictions_dict['df_val'] = pd.read_csv(join(output_foldername, 'df_val.csv'))
        predictions_dict['df_district'] = pd.read_csv(
            join(output_foldername, 'df_district.csv'))
        predictions_dict_list.append(predictions_dict)
    return predictions_dict_list

# Create combined predictions_dict
def _create_combined_pdict(predictions_dict_list, num_trials):
    predictions_dict_comb = {}
    predictions_dict_comb['plots'] = {}
    predictions_dict_comb['forecasts'] = {}
    # Copy all params
    predictions_dict_comb['run_params'] = copy.deepcopy(
        predictions_dict_list[0]['run_params'])
    # Copy all dataframes
    predictions_dict_comb['df_district'] = copy.deepcopy(
        predictions_dict_list[0]['df_district'])
    predictions_dict_comb['df_train'] = copy.deepcopy(
        predictions_dict_list[0]['df_train'])
    predictions_dict_comb['df_val'] = copy.deepcopy(
        predictions_dict_list[0]['df_val'])

    # Concatenate all predictions
    predictions_arr, losses_arr, params_arr = ([], [], [])
    for _, pdict in enumerate(predictions_dict_list):
        predictions_arr += pdict['trials']['predictions'][:num_trials]

        losses_arr.append(pdict['trials']['losses'][:num_trials])

        params_arr.append(pdict['trials']['params'][:num_trials])
    
    # Add them to predictions_dict_comb
    predictions_dict_comb['trials'] = {}
    predictions_dict_comb['trials']['predictions'] = predictions_arr
    predictions_dict_comb['trials']['losses'] = np.concatenate(
        tuple(losses_arr), axis=None)
    predictions_dict_comb['trials']['params'] = np.concatenate(
        tuple(params_arr), axis=None)

    return predictions_dict_comb


def _perform_beta_fitting(region_dict, config):
    uncertainty_args = {'predictions_dict': region_dict, 'fitting_config': config['fitting'],
                        **config['uncertainty']['uncertainty_params']}
    uncertainty = config['uncertainty']['method'](**uncertainty_args)

    uncertainty_forecasts = uncertainty.get_forecasts()
    for key in uncertainty_forecasts.keys():
        region_dict['forecasts'][key] = uncertainty_forecasts[key]['df_prediction']

    region_dict['forecasts']['ensemble_mean'] = uncertainty.ensemble_mean_forecast

    region_dict['plots']['beta_loss'], _ = plot_beta_loss(
        uncertainty.dict_of_trials)
    region_dict['beta'] = uncertainty.beta
    region_dict['beta_loss'] = uncertainty.beta_loss
    region_dict['deciles'] = uncertainty_forecasts
    
    return region_dict


def combine_multiple_runs(base_folder, list_of_pdicts, num_trials):
    # Create folder
    output_folder = join(dirname(base_folder), basename(base_folder) + '_comb')
    os.makedirs(output_folder, exist_ok=True)

    # Load config
    config = read_config(
        glob(f'{base_folder}/{list_of_pdicts[0]}/*.yaml')[0], abs_path=True)

    # Load and all pdicts and combine them
    predictions_dict_list = _load_all_pdicts(base_folder, list_of_pdicts)
    predictions_dict_comb = _create_combined_pdict(predictions_dict_list, num_trials)

    # Do the beta fitting
    config['uncertainty']['uncertainty_params']['fitting_method_params']['parallelise'] = False
    config['uncertainty']['uncertainty_params']['fitting_method_params']['n_jobs'] = 32
    predictions_dict_comb = _perform_beta_fitting(predictions_dict_comb, config=config)

    predictions_dict_comb['config'] = config

    return predictions_dict_comb

    # loc_dict_arr = Parallel(n_jobs=nthreads)(delayed(partial_perform_beta_fitting)(
    #     loc_dict) for _, loc_dict in tqdm(predictions_dict_comb.items()))

    # predictions_dict = dict(zip(predictions_dict_comb.keys(), loc_dict_arr))

    # with open(f'{output_folder}/predictions_dict.pkl', 'wb') as f:
    #     pickle.dump(predictions_dict, f)


def combine_multiple_runs_parallel(base_folder, pdicts_filename, num_trials, nthreads, parallelise=True):

    partial_combine_multiple_runs = partial(
        combine_multiple_runs, base_folder=base_folder, num_trials=num_trials)

    # Load list of pdicts file
    list_of_pdicts_mat = np.loadtxt(
        pdicts_filename, dtype='int', delimiter=' ').tolist()

    # Run function
    if parallelise:
        pdict_comb_arr = Parallel(n_jobs=nthreads)(delayed(partial_combine_multiple_runs)(
            **{'list_of_pdicts': list_of_pdicts}) for list_of_pdicts in tqdm(list_of_pdicts_mat))
    else:
        pdict_comb_arr = []
        for i, list_of_pdicts in enumerate(list_of_pdicts_mat):
           pdict_comb = partial_combine_multiple_runs(**{'list_of_pdicts': list_of_pdicts})
           pdict_comb_arr.append(pdict_comb)
           create_output(pdict_comb, dirname(base_folder), f'comb/{i}')


    # Save output
    for i, pdict in enumerate(pdict_comb_arr):
        create_output(pdict, dirname(base_folder), f'comb/{i}')

if __name__ == '__main__':

    # List of args
    parser = argparse.ArgumentParser(description="SEIR Batch Beta Fitting Script")
    parser.add_argument(
        "--base_folder", type=str, required=True, help="The folder where all the predictions_dict files have" +  
            " been saved in the format 0/, 1/, 2/, etc"
    )
    parser.add_argument(
        "--list_of_pdicts", nargs='+', type=str, required=False,
        help="The list of folder names from where the different trials need to be combined"
    )
    parser.add_argument(
        "--list_of_pdicts_fname", type=str, required=True,
        help="The fname of the txt file where the list of pdicts to be combined are stored"
    )
    parser.add_argument(
        "--nthreads", type=int, default=40, help="Number of threads to run in parallel"
    )
    parser.add_argument(
        "--num_trials", type=int, default=2000, help="Number of trials for each predictions_dict to concatenate"
    )
    parser.add_argument(
        "--parallelise", dest='parallelise', action='store_true', 
        help="All lists are run parallelly"
    )
    parser.add_argument(
        "--no-parallelise", dest='parallelise', action='store_false',
        help="All lists are run serially"
    )
    parsed_args = parser.parse_args()
    print(parsed_args)

    combine_multiple_runs_parallel(parsed_args.base_folder,
                                   parsed_args.list_of_pdicts_fname,
                                   parsed_args.num_trials,
                                   parsed_args.nthreads, 
                                   parsed_args.parallelise)

    print(f'Saved predictions_dict.')
            
