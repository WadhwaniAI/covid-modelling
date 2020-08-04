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

with open(f'../../misc/reports/{args.folder}/predictions_dict.pkl', 'rb') as pkl:
    region_dict = pickle.load(pkl)

create_report(region_dict, ROOT_DIR=f'../../misc/reports/{args.folder}/{t}')
region_dict['m1']['all_trials'].to_csv(f'../../misc/reports/{args.folder}/{t}/m1-trials.csv')
region_dict['m2']['all_trials'].to_csv(f'../../misc/reports/{args.folder}/{t}/m2-trials.csv')
region_dict['m2']['df_district'].to_csv(f'../../misc/reports/{args.folder}/true.csv')
region_dict['m2']['df_district_unsmoothed'].to_csv(f'../../misc/reports/{args.folder}/smoothed.csv')

