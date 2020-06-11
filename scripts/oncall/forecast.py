import pandas as pd

import pickle
import argparse
import copy
import json
import time

import sys
sys.path.append('../../')

from main.seir.forecast import create_decile_csv
from utils.create_report import create_report
from utils.enums import Columns
from viz import plot_ptiles, plot_r0_multipliers
from main.seir.uncertainty import get_all_ptiles, forecast_ptiles
from main.seir.forecast import save_r0_mul, predict_r0_multipliers

parser = argparse.ArgumentParser()
t = time.time()
parser.add_argument("-f", "--folder", help="folder name", required=True, type=str)
parser.add_argument("-i", "--iterations", help="number of trials for beta exploration", required=False, default=1500, type=int)
parser.add_argument("-date", "--date", help="date YYYY-MM-DD to sort trials by for cdf", required=True, type=str)
parser.add_argument("--fdays",help="how many days to forecast for", required=False, default=30, type=int)
args = parser.parse_args()   

with open(f'../../reports/{args.folder}/predictions_dict.pkl', 'rb') as pkl:
    region_dict = pickle.load(pkl)
forecast_dict = {}
df_reported = region_dict['m2']['df_district_unsmoothed']
date_of_interest = args.date

deciles_idx, forecast_dict['beta'] = get_all_ptiles(region_dict, date_of_interest, args.iterations)
with open(f'../../reports/{args.folder}/deciles_idx.json', 'w+') as params_json:
    deciles_idx['beta'] = forecast_dict['beta']
    json.dump(deciles_idx, params_json, indent=4)
    del deciles_idx['beta']

# forecast deciles to csv
forecast_dict['decile_params'], deciles_forecast = forecast_ptiles(region_dict, deciles_idx)
df_output = create_decile_csv(deciles_forecast, df_reported, region_dict['dist'], 'district', icu_fraction=0.02)
df_output.to_csv(f'../../reports/{args.folder}/deciles.csv')
pd.DataFrame(forecast_dict['decile_params']).to_csv(f'../../reports/{args.folder}/deciles-params.csv')

idxs = list(deciles_idx.values())
predictions = region_dict['m2']['predictions']
params = region_dict['m2']['params']
losses = region_dict['m2']['losses']
def get_idxs(l):
    return {ptile : l[deciles_idx[ptile]] for ptile in deciles_idx.keys()}
plots = plot_ptiles(region_dict, get_idxs(predictions), vline=date_of_interest)
plots[Columns.active].figure.savefig(f'../../reports/{args.folder}/deciles.png')
forecast_dict['deciles_plot'] = plots[Columns.active]

# perform what-ifs on 80th percentile
predictions_mul_dict = predict_r0_multipliers(region_dict, params[deciles_idx[80]], days=args.fdays)
save_r0_mul(predictions_mul_dict, folder=args.folder)

forecast_dict['what-ifs-plot'] = plot_r0_multipliers(region_dict, params[deciles_idx[80]], predictions_mul_dict, multipliers=[0.9, 1, 1.1, 1.25])
forecast_dict['what-ifs-plot'].figure.savefig(f'../../reports/{args.folder}/what-ifs/what-ifs.png')

create_report(region_dict, forecast_dict=forecast_dict, ROOT_DIR=f'../../reports/{args.folder}')
print(f"lit. done: view files at ../../reports/{args.folder}/")
