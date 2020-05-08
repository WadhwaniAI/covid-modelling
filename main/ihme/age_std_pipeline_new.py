import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import curvefit
from curvefit.core.functions import *

sys.path.append('../..')
from utils.age_standardisation import standardise_age, census_age
from models.ihme.pipeline import WAIPipeline
from models.ihme.util import Params
from models.ihme.data import get_district_timeseries_cached

parser = argparse.ArgumentParser() 
parser.add_argument("-d", "--district", help="district name", required=True)
parser.add_argument("-l", "--log", help="fit on log", required=False, action='store_true')
parser.add_argument("-a", "--age", help="perform age-standardisation", required=False, action='store_true')
parser.add_argument("-sd", "--sd", help="use social distance covariate", required=False, action='store_true')
args = parser.parse_args()

age_data = census_age()

amd_census_name = ['District - Ahmadabad (07)']
mumbai_census_name = ['District - Mumbai (23)', 'District - Mumbai Suburban (22)']
pune_census_name = ['District - Pune (25)']
delhi_census_name = ['State - NCT OF DELHI (07)']
jaipur_census_name = ['District - Jaipur (12)']
bengaluru_census_name = ['District - Bangalore (18)', 'District - Bangalore Rural (29)']

# tuples
mumbai = 'Mumbai', 'Maharashtra', mumbai_census_name
amd = 'Ahmedabad', 'Gujarat', amd_census_name
jaipur = 'Jaipur', 'Rajasthan', jaipur_census_name
pune = 'Pune', 'Maharashtra', pune_census_name
delhi = 'all', 'Delhi', delhi_census_name
bengaluru = ['Bengaluru Urban', 'Bengaluru Rural'], 'Karnataka', bengaluru_census_name

cities = {
    'mumbai': mumbai,
    'ahmedabad': amd,
    'jaipur': jaipur,
    'pune': pune,
    'delhi': delhi,
    'bengaluru': bengaluru,
}

# -------------------

def run_pipeline(triple):
    dist, st, area_names = triple
    fname = f'age_std_{dist}{st}_deaths'
    district_timeseries = get_district_timeseries_cached(dist, st)
    df, dtp = standardise_age(district_timeseries, age_data, dist, st, area_names)
    df = df.reset_index(col_fill='date')

    ycol = 'age_std' if args.age else 'non_std'
    if args.log:
        obs = 'log_non_std'
        func = log_erf
        ycol = f'log_{ycol}' 
        label = 'log_age_std'
    else:
        obs = 'non_std'
        func = erf
        label = 'age_std'

    df = df.loc[df['age_std'].gt(1e-15).idxmax():,:]
    df['day'] = [x for x in range(len(df))]
    df['date']= pd.to_datetime(df['date']) 
    df['log_age_std'] = df['age_std'].apply(np.log)
    df['log_non_std'] = df['non_std'].apply(np.log)
    df.loc[:,'group'] = len(df) * [ 1.0 ]
    df.loc[:,'covs'] = len(df) * [ 1.0 ]
    df.loc[:,'sd'] = df['date'].apply(lambda x: [1.0 if x >= datetime(2020, 3, 24) else 0.0]).tolist()
    df.loc[:,f'{ycol}_normalized'] = df[ycol]/df[ycol].max()

    params = Params.fromdefault(df, {
        'groupcol': 'group',
        'xcol': 'day',
        'ycols': {
            ycol: func,
        }
    }, default_label=label)

    # set vars

    # output
    today = datetime.today()
    output_folder = f'output/pipeline/{fname}/{today}'
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    if args.sd:
        covs = ['covs', 'sd', 'covs']
    else:
        covs = ['covs', 'covs', 'covs']

    pipeline = WAIPipeline(df, ycol, params, covs, fname)
    pipeline.run()
    print(pipeline.pipeline.mod.params)
    p, _ = pipeline.predict()

    if args.log:
        predicted_cumulative_deaths = pipeline.lograte_to_cumulative(p, dtp, output_folder)
    else:
        predicted_cumulative_deaths = pipeline.rate_to_cumulative(p, dtp, output_folder)

    pipeline._plot_draws(pipeline.pipeline.fun)
    plt.savefig(f'{output_folder}/{pipeline.file_prefix}_{pipeline.pipeline.col_obs}_{pipeline.func.__name__}_draws.png')
    plt.clf()
    pipeline.plot_results(predicted_cumulative_deaths, ycol='cumulative')
    if obs != ycol:
        plt.plot(df['date'], df[obs], 'b+', label='observed (non-standardized)')
    plt.savefig(f'{output_folder}/{pipeline.file_prefix}_{pipeline.pipeline.col_obs}_{pipeline.func.__name__}.png')
    plt.clf()

# -------------------
run_pipeline(cities[args.district])
# -------------------