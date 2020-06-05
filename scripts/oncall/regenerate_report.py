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

create_report(region_dict, ROOT_DIR=f'../../reports/{args.folder}/{t}') 
