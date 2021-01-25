import argparse
import copy
import json
import os
import sys
from datetime import datetime, timedelta

from curvefit.core import functions

sys.path.append('../..')

from data.processing import jhu
from utils.fitting.population import standardise_age

from utils.fitting.data import lograte_to_cumulative, rate_to_cumulative, regions
from main.ihme.fitting import run_cycle, create_output_folder
from utils.fitting.util import train_test_split, read_config
# -------------------

def find_init(country, triple):
    country = country
    indian_district, indian_state, area_names = triple

    timeseries = jhu(country.title())
    df, dtp = standardise_age(timeseries, country, indian_state, area_names)

    label = 'default' if args.log else 'rate'
    _ , params = read_config(f'config/{label}.yaml')

    ycol = 'age_std_mortality'
    params['ycol'] = f'log_{ycol}' if args.log else ycol
    
    # set vars
    df.loc[:,'sd'] = df['date'].apply(lambda x: [1.0 if x >= datetime(2020, 3, 24) else 0.0]).tolist()
    params['covs'] = ['covs', 'sd', 'covs'] if args.sd else ['covs', 'covs', 'covs']
    
    if args.smoothing:
        print(f'smoothing {args.smoothing}')
        smoothedcol = f'{ycol}_smoothed'
        df[smoothedcol] = df[ycol].rolling(args.smoothing).mean()

    df = df.loc[df['mortality'].gt(1e-15).idxmax():,:]
    
    # output
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_folder = create_output_folder(f'age_std/{country}-{indian_district}/{now}')
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    with open(f'{output_folder}/params.json', 'w') as pfile:
        pargs = copy.copy(params)
        pargs['sd'] = args.sd
        pargs['smoothing'] = args.smoothing
        pargs['log'] = args.log
        json.dump(pargs, pfile, indent=4)

    params['func'] = getattr(functions, params['func'])
    xform_func = lograte_to_cumulative if args.log else rate_to_cumulative
    test_size=7
    train, test = train_test_split(df, threshold=df['date'].max() - timedelta(days=test_size))
    dataframes = {
        'df': df,
        'train': train,
        'test': test,
    }
    results = run_cycle(dataframes, params, dtp=dtp, xform_func=xform_func, max_evals=args.max_evals)
    trials_dict = results['trials']
    best_loss, best_params = float('inf'), None
    for run in trials_dict.keys():
        trials = trials_dict[run]
        if trials.best_trial['result']['loss'] < best_loss:
            best_loss = trials.best_trial['result']['loss']
            best_params = trials.best_trial['misc']['vals']
    print("best: {}".format(best_params))
    print("best err: {}".format(best_loss))

# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-p", "--place", help="place to fit to", required=True)
    parser.add_argument("-d", "--district", help="indian district name to standardise to", required=True)
    parser.add_argument("-l", "--log", help="fit on log", required=False, action='store_true')
    parser.add_argument("-sd", "--sd", help="use social distance covariate", required=False, action='store_true')
    parser.add_argument("-s", "--smoothing", help="how much to smooth, else no smoothing", required=False, type=int)
    parser.add_argument("-i", "--max_evals", help="max evals on each hyperopt run", required=False, default=50, type=int)
    args = parser.parse_args()
    
    find_init(args.place, regions[args.district])
# -------------------