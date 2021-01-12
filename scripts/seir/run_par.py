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
from main.seir.sensitivity import calculate_sensitivity_and_plot
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
config_filename = 'uncer.yaml'


# %%
config = read_config(config_filename)
prediction_dict = {}
prediction_dict['m1'] = single_fitting_cycle(**copy.deepcopy(config['fitting']))                                                      

# %%
import pickle as pkl
with open('../../misc/predictions/exp_set2.pickle', 'wb') as handle:
    pkl.dump(prediction_dict, handle)
