import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append('../..')
from models.ihme.pipeline import WAIPipeline
from models.ihme.params import Params


parser = argparse.ArgumentParser() 
parser.add_argument("-p", "--params", help="name of entry in params.json", required=True)
parser.add_argument("-d", "--daily", help="whether or not to plot daily", required=False, action='store_true')
parser.add_argument("-s", "--smoothing", help="how much to smooth, else no smoothing", required=False)
args = parser.parse_args() 

# load params
daily, smoothing_window = args.daily, args.smoothing
params = Params.fromjson(args.params)

# assign data
df = params.df
deriv = True if daily else False

# output
fname = args.params
today = datetime.today()
output_folder = f'output/deaths/{fname}/{today}'
if not os.path.exists(output_folder):
        os.makedirs(output_folder)

with open(f'{output_folder}/params.json', 'w') as pfile:
    json.dump(params.pargs, pfile)


for i, (ycol, func) in enumerate(params.ycols.items()):
    df.loc[:,'covs'] = len(df) * [ 1.0 ]
    df.loc[:,'sd'] = df[params.date].apply(lambda x: [1.0 if x >= datetime(2020, 3, 24) else 0.0]).tolist()
    df.loc[:,f'{ycol}_normalized'] = df[ycol]/df[ycol].max()
    covs = ['covs', 'covs', 'covs']

    pipeline = WAIPipeline(df, ycol, params, covs, file_prefix='', smoothing=smoothing_window, predict_space=None)
    pipeline.run()
    p, gp = pipeline.predict()
    pipeline.all_plots(output_folder, p, deriv=deriv)