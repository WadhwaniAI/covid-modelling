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
from models.ihme.util import cities
from utils.util import read_config
from main.seir.forecast import get_forecast, order_trials, top_k_trials, forecast_k
from utils.enums import Columns

from main.ihme.fitting import single_cycle, create_output_folder
from utils.enums import Columns
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

# m1
m1_results = single_cycle(dist, st, area_names, copy(model_params), which_compartments=Columns.which_compartments(), **config)

df_true = m1_results['df_district']
df_pred = m1_results['df_prediction']

makesum = copy(df_pred)
makesum['total_infected'] = df_pred['recovered'] + df_pred['deceased'] + df_pred['hospitalised']

plot_forecast(df_true, makesum.reset_index(), model_name='IHME M1', dist=dist, state=st, filename=os.path.join(output_folder, 'm1.png'))

# m2
m2_config = copy(config)
m2_config['test_size'] = 0
m2_results = single_cycle(dist, st, area_names, copy(model_params), which_compartments=Columns.which_compartments(), **m2_config)

df_true = m2_results['df_district']
df_pred = m2_results['df_prediction']

makesum = copy(df_pred)
makesum['total_infected'] = df_pred['recovered'] + df_pred['deceased'] + df_pred['hospitalised']

plot_forecast(df_true, makesum.reset_index(), model_name='IHME M2', dist=dist, state=st, filename=os.path.join(output_folder, 'm2.png'))

runtime = time.time() - start_time
print('time:', runtime)

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