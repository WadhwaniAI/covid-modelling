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
predictions_dict = {}
config_filename = 'exp_simulate_1.yaml'
print(config_filename)
config = read_config(config_filename)

predictions_dict['m1'] = single_fitting_cycle(**copy.deepcopy(config['fitting']))                                           

# %%
import pickle as pkl
with open('../../misc/predictions/exp_simulate_1_70.pickle', 'wb') as handle:
    pkl.dump(predictions_dict, handle)
# %%
