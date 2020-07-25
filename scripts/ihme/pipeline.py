import json
import sys
import time
from copy import copy
from datetime import datetime

import dill as pickle
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('../..')
from utils.data import regions

from viz import plot_ihme_results
from main.ihme.fitting import create_output_folder, single_cycle
from utils.util import read_config
from utils.enums import Columns

import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', module='pandas', category=RuntimeWarning) #, message='invalid value encountered in')
warnings.filterwarnings('ignore', module='curvefit', category=RuntimeWarning) #, message='invalid value encountered in')

val_size = 7
test_size = 7
min_days = 7
scoring = 'mape'
# -------------------

def run_pipeline(dist, st, area_names, config, model_params, folder):
    start_time = time.time()
    ycol, xform_col = model_params['ycol'], '{log}{colname}_rate'.format(log='log_' if config['log'] else '', colname=model_params['ycol'])
    results_dict = single_cycle(dist, st, area_names, model_params, which_compartments=[Columns.deceased], **config)
    output_folder = create_output_folder(f'forecast/{folder}')
    predictions = results_dict['df_prediction']
    train, test, df = results_dict['df_train'], results_dict['df_val'], results_dict['df_district']
    runtime = time.time() - start_time
    print('runtime:', runtime)
    # PLOTTING
    plot_df, plot_test = copy(df), copy(test)
    plot_df[ycol] = plot_df[ycol]
    plot_test[ycol] = plot_test[xform_col]

    plot_ihme_results(model_params, results_dict['mod.params'], plot_df, len(train), plot_test, predictions[ycol], 
        predictions['date'], results_dict['df_loss']['val'], dist, val_size, draws=results_dict['draws'][ycol]['draws'], yaxis_name='cumulative deaths')
    plt.savefig(f'{output_folder}/results.png')
    plt.clf()

    # SAVE PARAMS INFO
    with open(f'{output_folder}/params.json', 'w') as pfile:
        pargs = copy(model_params)
        pargs.update(config)
        pargs['func'] = pargs['func'].__name__
        pargs['priors']['fe_init'] = results_dict['best_params'][ycol]
        pargs['n_days_train'] = int(results_dict['n_days'][ycol])
        pargs['error'] = results_dict['df_loss'].to_dict()
        pargs['runtime'] = runtime
        json.dump(pargs, pfile, indent=4)

    # SAVE DATA, PREDICTIONS
    picklefn = f'{output_folder}/data.pkl'
    with open(picklefn, 'wb') as pickle_file:
        pickle.dump(results_dict, pickle_file)
    
    print(f"yee we done see results here: {output_folder}")

# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-d", "--district", help="district name", required=True)
    parser.add_argument("-c", "--config", help="config file name", required=True)
    parser.add_argument("-f", "--folder", help="folder name", required=False, default=None, type=str)
    args = parser.parse_args()
    
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = f'{args.district}/{str(now)}' if args.folder is None else args.folder
    
    config, model_params = read_config(args.config)
    dist, st, area_names = regions[args.district]
    run_pipeline(dist, st, area_names, config, model_params, folder)

# -------------------