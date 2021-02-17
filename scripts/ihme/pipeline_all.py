import os
import json
from copy import copy
import pandas as pd
import numpy as np
import dill as pickle
import time
import argparse
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from adjustText import adjust_text

import sys
sys.path.append('../..')
from utils.fitting.data import cities
from utils.generic.config import read_config
from main.seir.forecast import get_forecast, order_trials, top_k_trials, forecast_k
from utils.generic.enums import Columns

from main.ihme.fitting import single_cycle, create_output_folder
from utils.generic.enums import Columns
from viz.fit import plot_fit
from viz.forecast import plot_forecast_agnostic as plot_forecast


parser = argparse.ArgumentParser() 
parser.add_argument("-d", "--district", help="district name", required=True)
parser.add_argument("-c", "--config", help="config file name", required=True)
parser.add_argument("-f", "--folder", help="folder name", required=False, default=None, type=str)
args = parser.parse_args()
config, model_params = read_config(args.config)
dist, st, area_names = cities[args.district]
now = datetime.now().strftime("%Y%m%d-%H%M%S")
folder = f'{args.district}/{str(now)}' if args.folder is None else args.folder
output_folder = create_output_folder(f'forecast/{folder}')

start_time = time.time()

which_compartments = Columns.curve_fit_compartments()
# 
m1_results = single_cycle(dist, st, area_names, copy(model_params), which_compartments=which_compartments, **config)
df_train, df_val = m1_results['df_train'], m1_results['df_val']
df_train_nora, df_val_nora = m1_results['df_train_nora'], m1_results['df_val_nora']
df_true = m1_results['df_district']
df_pred = m1_results['df_prediction']

makesum = copy(df_pred)
makesum['total'] = df_pred['recovered'] + df_pred['deceased'] + df_pred['active']

plot_fit(
    makesum.reset_index(), df_train, df_val, df_train_nora, df_val_nora, 
    m1_results['n_days'][which_compartments[0].name], st, dist, which_compartments=[c.name for c in which_compartments],
    savepath=os.path.join(output_folder, 'm1.png'))
plot_forecast(df_true, makesum.reset_index(), model_name='IHME M1', dist=dist, state=st, filename=os.path.join(output_folder, 'm1-forecast.png'))

# m2
m2_config = copy(config)
m2_config['test_size'] = 0
m2_results = single_cycle(dist, st, area_names, copy(model_params), which_compartments=which_compartments, **m2_config)

df_train, df_val = m2_results['df_train'], m2_results['df_val']
df_train_nora, df_val_nora = m2_results['df_train_nora'], m2_results['df_val_nora']
df_true = m2_results['df_district']
df_pred = m2_results['df_prediction']

makesum = copy(df_pred)
makesum['total'] = df_pred['recovered'] + df_pred['deceased'] + df_pred['active']

plot_fit(
    makesum.reset_index(), df_train, df_val, df_train_nora, df_val_nora, 
    m2_results['n_days'][which_compartments[0].name], st, dist, which_compartments=[c.name for c in which_compartments],
    savepath=os.path.join(output_folder, 'm2.png'))
plot_forecast(df_true, makesum.reset_index(), model_name='IHME M2', dist=dist, state=st, filename=os.path.join(output_folder, 'm2-forecast.png'))

runtime = time.time() - start_time
print('time:', runtime)

# SAVE LOSS
with open(f'{output_folder}/loss.json', 'w') as pfile:
    l = {
        'm1': m1_results['df_loss'].to_dict(),
        'm2': m2_results['df_loss'].to_dict(),
    }
    json.dump(l, pfile)

m1_results['df_prediction'].to_csv(os.path.join(output_folder, 'm1-pred.csv'))
m2_results['df_prediction'].to_csv(os.path.join(output_folder, 'm2-pred.csv'))

# SAVE CONFIG
with open(f'{output_folder}/params.json', 'w') as pfile:
    pargs = copy(model_params)
    pargs.update(config)
    # pargs['func'] = pargs['func'].__name__
    json.dump(pargs, pfile)

# SAVE DATA, PREDICTIONS
picklefn = f'{output_folder}/data.pkl'
with open(picklefn, 'wb') as pickle_file:
    pickle.dump({'m1': m1_results, 'm2': m2_results}, pickle_file)

print(f"yee we done see results here: {output_folder}")