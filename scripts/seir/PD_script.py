# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython



# %%
import numpy as np
import matplotlib.pyplot as plt
import datetime
import copy
import time
import os
import sys
sys.path.append('../../')

from data.processing import get_data

import models

from main.seir.fitting import single_fitting_cycle
from main.seir.forecast import get_forecast, forecast_all_trials, create_all_trials_csv, create_decile_csv_new
from utils.generic.create_report import save_dict_and_create_report
from utils.generic.config import read_config
from utils.generic.enums import Columns
from utils.fitting.loss import Loss_Calculator
#from utils.generic.logging import log_wandb
from viz import plot_forecast, plot_top_k_trials, plot_ptiles

import yaml


# %%
from os.path import exists, join, splitext


# %%
models = {'MCMC':'uncer.yaml','BO':'default.yaml'}
# models = {'BO':'default.yaml'}


# %%
runs = 10
PD = {}
for n in range(runs):
    print("The iteration number is ",n)
    print("The iteration number is ",n)
    PD[f'm{n}']= {}
    for model_name,config_filename in models.items():
        config = read_config(config_filename)
        prediction_dict = {}
        prediction_dict['m1'] = single_fitting_cycle(**copy.deepcopy(config['fitting']))
        prediction_dict['m1']['forecasts'] = {}
        prediction_dict['m1']['forecasts']['best'] = get_forecast(prediction_dict, train_fit='m1', 
                                                                model=config['fitting']['model'], 
                                                                forecast_days=config['forecast']['forecast_days'])

        prediction_dict['m1']['trials_processed'] = forecast_all_trials(prediction_dict, train_fit='m1', 
                                                                        model=config['fitting']['model'], 
                                                                        forecast_days=config['forecast']['forecast_days'])
        uncertainty_args = {'predictions_dict': prediction_dict, 'fitting_config': config['fitting'],
                    'forecast_config': config['forecast'], **config['uncertainty']['uncertainty_params']}
        uncertainty = config['uncertainty']['method'](**uncertainty_args)
        prediction_dict['uncertainty_forecasts'] = uncertainty.get_forecasts()
        prediction_dict['ensemble_mean_forecast'] = uncertainty.ensemble_mean_forecast
        prediction_dict['m1']['metric']['p_val'] = uncertainty.p_val
        PD[f'm{n}'][model_name] = prediction_dict                                                               

# %%
import pickle as pkl
with open('../../misc/predictions/test_mumbai_logdiff.pickle', 'wb') as handle:
    pkl.dump(PD, handle)
