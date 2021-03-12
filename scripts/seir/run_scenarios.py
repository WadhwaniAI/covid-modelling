import numpy as np
import matplotlib.pyplot as plt
import datetime
import copy
import time

import sys
sys.path.append('../../')

import models

from main.seir.fitting import single_fitting_cycle
from main.seir.forecast import get_forecast, forecast_all_trials, create_all_trials_csv, create_decile_csv_new
from utils.generic.create_report import save_dict_and_create_report
from utils.generic.config import read_config, make_date_key_str
from utils.generic.enums import Columns
from utils.fitting.loss import Loss_Calculator
from utils.generic.logging import log_wandb, log_mlflow
from viz import plot_forecast, plot_top_k_trials, plot_ptiles

import yaml
import wandb
config_filenames = ['exp_simulate_3.yaml']
train_periods = [5,11,17,23,28]
PD = {}

for i,config_filename in enumerate(config_filenames):
    config = read_config(config_filename)
    PD[i] = {}
    x = 1
    for j,train_period in enumerate(train_periods):
        
        config['fitting']['split']['train_period'] = train_period
        print(config_filename,'-RUN------------------------->',x," ...Period-",config['fitting']['split']['train_period'])
        x = x+1
        PD[i][j] = single_fitting_cycle(**copy.deepcopy(config['fitting']))                                          


import pickle as pkl
with open('../../misc/predictions/PD_beta_TP.pickle', 'wb') as handle:
    pkl.dump(PD, handle)
