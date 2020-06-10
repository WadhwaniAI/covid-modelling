import os
import sys
import json
import copy
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import curvefit
from curvefit.core.functions import *

sys.path.append('../..')
from models.ihme.pipeline import WAIPipeline
from models.ihme.util import get_mortality
from models.ihme.params import Params
from data.processing import jhu
from models.ihme.population import standardise_age
from main.ihme.mortality_pipeline import cities

# -------------------

def find_init(country, triple):
    country = country
    indian_district, indian_state, area_names = triple
    fname = f'{country}_deaths_age_std_{indian_district}'

    timeseries = jhu(country.title())
    df, dtp = standardise_age(timeseries, country, indian_state, area_names)
    ycol = 'age_std_mortality'
    label = 'log_mortality' if args.log else 'mortality'
    func = log_erf if args.log else erf

    df[f'log_{ycol}'] = df[ycol].apply(np.log)
    ycol = f'log_{ycol}' if args.log else ycol
    
    # set vars
    df['day'] = [x for x in range(len(df))]
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
    
    # output
    today = datetime.today()
    output_folder = f'output/mortality/{fname}/{today}'
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    with open(f'{output_folder}/params.json', 'w') as pfile:
        pargs = copy.copy(params.pargs)
        pargs['sd'] = args.sd
        pargs['smoothing'] = args.smoothing
        pargs['log'] = args.log
        pargs['ycols'] = {k: v.__name__ for k, v in pargs['ycols'].items()}
        json.dump(pargs, pfile)

    best = {}
    bounds = params.pargs['priors']['fe_bounds']
    step = 5
    all_inits = [[i, j, k] for i in range(bounds[0][0], bounds[0][1], step)  
                for j in range(bounds[1][0], bounds[1][1], step) 
                for k in range(bounds[2][0], bounds[2][1], step)]
    print(len(all_inits))
    for inits in all_inits:
        params = Params.fromdefault(df, {
            'ycols': {
                ycol: func
            },
            'priors': {
                'fe_init': inits,
                'fe_bounds': params.pargs['priors']['fe_bounds']
            }
        }, default_label=label)
        pipeline = WAIPipeline(df, ycol, params, covs, fname)
        pipeline.run()
        p, _ = pipeline.predict()
        best[pipeline.calc_error(p)['overall']['rmse']] = inits
    print (best)
    print("best: {}".format(best[min(best.keys())]))
    print("best err: {}".format(min(best.keys())))

# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-p", "--place", help="place to fit to", required=True)
    parser.add_argument("-d", "--district", help="indian district name to standardise to", required=True)
    parser.add_argument("-l", "--log", help="fit on log", required=False, action='store_true')
    parser.add_argument("-sd", "--sd", help="use social distance covariate", required=False, action='store_true')
    parser.add_argument("-s", "--smoothing", help="how much to smooth, else no smoothing", required=False)
    args = parser.parse_args()
    
    find_init(args.place, cities[args.district])
# -------------------