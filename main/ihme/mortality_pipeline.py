import os
import sys
import json
import copy
import random
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sigfig import round
import curvefit
from curvefit.core.functions import *

sys.path.append('../..')
from models.ihme.pipeline import WAIPipeline
from models.ihme.util import get_mortality
from models.ihme.params import Params
from models.ihme.dataloader import get_district_timeseries_cached
pd.options.mode.chained_assignment = None

# tuples: (district, state, census_area_name(s))
mumbai = 'Mumbai', 'Maharashtra', ['District - Mumbai (23)', 'District - Mumbai Suburban (22)']
amd = 'Ahmedabad', 'Gujarat', ['District - Ahmadabad (07)']
jaipur = 'Jaipur', 'Rajasthan', ['District - Jaipur (12)']
pune = 'Pune', 'Maharashtra', ['District - Pune (25)']
delhi = 'Delhi', 'Delhi', ['State - NCT OF DELHI (07)']
bengaluru = 'Bengaluru', 'Karnataka', ['District - Bangalore (18)', 'District - Bangalore Rural (29)']

cities = {
    'mumbai': mumbai,
    'ahmedabad': amd,
    'jaipur': jaipur,
    'pune': pune,
    'delhi': delhi,
    'bengaluru': bengaluru,
}

# -------------------

def run_pipeline(triple, args):
    dist, st, area_names = triple
    fname = f'{dist}_deaths'

    district_timeseries = get_district_timeseries_cached(dist, st)
    df, dtp = get_mortality(district_timeseries, st, area_names)

    label = f'log_mortality' if args.log else 'mortality'
    ycol = f'log_mortality' if args.log else 'mortality'
    func = log_erf if args.log else erf

    # set vars
    df['date']= pd.to_datetime(df['date'])
    df.loc[:,'group'] = len(df) * [ 1.0 ]
    df.loc[:,'covs'] = len(df) * [ 1.0 ]
    df.loc[:,'sd'] = df['date'].apply(lambda x: [1.0 if x >= datetime(2020, 3, 24) else 0.0]).tolist()
    covs = ['covs', 'sd', 'covs'] if args.sd else ['covs', 'covs', 'covs']

    if args.smoothing:
        print(f'smoothing {args.smoothing}')
        smoothedcol = f'{ycol}_smoothed'
        df[smoothedcol] = df[ycol].rolling(args.smoothing).mean()
        ycol = smoothedcol
    
    params = Params.fromdefault(df, {
        'ycols': {
            ycol: func
        }
    }, default_label=label)
    
    startday = df['date'][df['mortality'].gt(1e-15).idxmax()]
    df = df.loc[df['mortality'].gt(1e-15).idxmax():,:]
    df.loc[:, 'day'] = (df['date'] - np.min(df['date'])).apply(lambda x: x.days)
    # output
    today = datetime.today()
    output_folder = f'output/mortality/{fname}/{today}'
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def rp(params, pf='', priors=None):
        pipeline = WAIPipeline(df, ycol, params, covs, fname, priors=priors)
        pipeline.run()
        print(pipeline.pipeline.mod.params)
        print(pipeline.pipeline.fit_dict)
        p, _ = pipeline.predict()
        err = pipeline.calc_error(p)

        low, up = pipeline._plot_draws(pipeline.pipeline.fun)
        draws = np.vstack((low, up))
        
        if args.log:
            predicted_cumulative_deaths, draws = pipeline.lograte_to_cumulative(p, dtp, draws=draws)
        else:
            predicted_cumulative_deaths, draws = pipeline.rate_to_cumulative(p, dtp, draws=draws)

        plt.savefig(f'{output_folder}/{pf}{pipeline.file_prefix}_{pipeline.pipeline.col_obs}_{pipeline.func.__name__}_draws.png')
        plt.clf()
        pipeline.plot_results(predicted_cumulative_deaths, ycol='cumulative', draws=draws)
        plt.axvline(startday, ls=':', c='cadetblue', label='train start boundary')
        plt.legend()
        plt.savefig(f'{output_folder}/{pf}{pipeline.file_prefix}_{pipeline.pipeline.col_obs}_{pipeline.func.__name__}.png')
        plt.clf()
        return err

    if args.search:
        def randomSearch(all_inits, iterations, bounds, params, scoring='rmse', seed=None):
            if seed is None:
                seed = datetime.today().timestamp()
            random.seed(seed)
            indices = random.sample(range(len(all_inits)), iterations)
            min_test_err, min_train_err = np.inf, np.inf
            best_init_test, best_init_train = None, None
            for idx in indices:
                priors = {
                    'fe_init': all_inits[idx],
                    'fe_bounds': bounds
                }
                pipeline = WAIPipeline(df, ycol, params, covs, fname, priors=priors)
                pipeline.run()
                p, _ = pipeline.predict()
                err = pipeline.calc_error(p)
                rounded_train_err = round(float(err['train'][scoring]), sigfigs=5)
                rounded_test_err = round(float(err['test'][scoring]), sigfigs=5)
                if err['train'][scoring] < min_train_err:
                    min_train_err = rounded_train_err
                    best_init_train = [all_inits[idx]]
                elif min_train_err == rounded_train_err:
                    best_init_train.append(all_inits[idx])
                if err['test'][scoring] < min_test_err:
                    min_test_err = rounded_test_err
                    best_init_test = [all_inits[idx]]
                elif min_test_err == rounded_test_err:
                    best_init_test.append(all_inits[idx])

            return (min_train_err, best_init_train), (min_test_err, best_init_test)

        bounds = params.pargs['priors']['fe_bounds']
        step = (0.1, 2, 0.5)
        all_inits = [[float(i),float(j),float(k)] for i in np.arange(bounds[0][0], bounds[0][1], step[0])
                                for j in np.arange(bounds[1][0], bounds[1][1], step[1])
                                for k in np.arange(bounds[2][0], bounds[2][1], step[2])]
        (min_train_err, best_init_train), (min_test_err, best_init_test) = \
            randomSearch(all_inits, int(args.search), bounds, params)
        # print(f'train: ({min_train_err}, {best_init_train[0]})')
        print(f'test: ({min_test_err}, {best_init_test})')

        best_priors = {
            'fe_init': best_init_test[0],
            'fe_bounds': bounds
        }
        with open(f'{output_folder}/randomsearch.json', 'w') as rsfile:
            out = {
                'args.search': args.search,
                'train': {
                    'min_err': min_train_err,
                    'best_init': best_init_train,
                },
                'test': {
                    'min_err': min_test_err,
                    'best_init': best_init_test,
                }
            }
            json.dump(out, rsfile)

        err = rp(params, pf=f'best_test_rmse', priors=best_priors)
        with open(f'{output_folder}/params.json', 'w') as pfile:
            pargs = copy.copy(params.pargs)
            pargs['ycols'] = {k: v.__name__ for k, v in pargs['ycols'].items()}
            pargs['priors'] = best_priors
            pargs['error'] = err
            pargs['sd'] = args.sd
            pargs['smoothing'] = args.smoothing
            pargs['log'] = args.log
            pargs['search_iterations'] = args.search
            json.dump(pargs, pfile)       
    else:
        err = rp(params)
        with open(f'{output_folder}/params.json', 'w') as pfile:
            pargs = copy.copy(params.pargs)
            pargs['ycols'] = {k: v.__name__ for k, v in pargs['ycols'].items()}
            pargs['error'] = err
            pargs['sd'] = args.sd
            pargs['smoothing'] = args.smoothing
            pargs['log'] = args.log
            json.dump(pargs, pfile)

# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-d", "--district", help="district name", required=True)
    parser.add_argument("-l", "--log", help="fit on log", required=False, action='store_true')
    parser.add_argument("-sd", "--sd", help="use social distance covariate", required=False, action='store_true')
    parser.add_argument("-s", "--smoothing", help="how much to smooth, else no smoothing", required=False)
    parser.add_argument("-rs", "--search", help="whether to do randomsearch", required=False)
    args = parser.parse_args()

    run_pipeline(cities[args.district], args)
# -------------------