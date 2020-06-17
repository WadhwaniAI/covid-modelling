import time
import argparse
import pickle

import sys
sys.path.append('../../')

from utils.create_report import create_report

'''
Regenerates report from pickled file
ex. 
python3 regenerate_report.py -f dirtoreport
python3 regenerate_report.py -f dirtoreport
'''

# --- turn into command line args
parser = argparse.ArgumentParser()
t = time.time()
parser.add_argument("-f", "--folder", help="folder name", required=True, type=str)
args = parser.parse_args()

with open(f'../../reports/{args.folder}/predictions_dict.pkl', 'rb') as pkl:
    region_dict = pickle.load(pkl)

from utils.create_report import trials_to_df
from main.seir.forecast import get_all_trials

predictions, losses, params = get_all_trials(region_dict, train_fit='m1')
region_dict['m1']['params'] = params
region_dict['m1']['losses'] = losses
region_dict['m1']['predictions'] = predictions
region_dict['m1']['all_trials'] = trials_to_df(predictions, losses, params)
predictions, losses, params = get_all_trials(region_dict, train_fit='m2')
region_dict['m2']['params'] = params
region_dict['m2']['losses'] = losses
region_dict['m2']['predictions'] = predictions
region_dict['m2']['all_trials'] = trials_to_df(predictions, losses, params)

import os
if not os.path.exists(f'../../reports/{args.folder}/{t}'):
    os.makedirs(f'../../reports/{args.folder}/{t}')

region_dict['m1']['all_trials'].to_csv(f'../../reports/{args.folder}/{t}/m1-trials.csv')
region_dict['m2']['all_trials'].to_csv(f'../../reports/{args.folder}/{t}/m2-trials.csv')
region_dict['m1']['df_district'].to_csv(f'../../reports/{args.folder}/{t}/true.csv')

