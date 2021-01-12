import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
import datetime
import copy
import time
import wandb
import pickle as pkl
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
from utils.generic.logging import log_wandb
import yaml
from data.dataloader import SimulatedDataLoader
from joblib import delayed, Parallel
import os

# config_file, scenario_name

config_filename = sys.argv[1]
config = read_config(config_filename)

scenario_name = sys.argv[2]
scenario_dict = {}

def run_bo():
    predictions_dict = single_fitting_cycle(**copy.deepcopy(config['fitting']))
    output = predictions_dict['best_params']
    output.update(predictions_dict['default_params'])
    return output

best_fit_params = []
best_fit_params = Parallel(n_jobs=4)(
    delayed(run_bo)() for _ in range(4)
)

scenario_dict[scenario_name] = best_fit_params
save_dir = '../../misc/predictions/deg_exp/'    
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(os.path.join(save_dir, scenario_name + ".pickle"), 'wb') as handle:
    pkl.dump(scenario_dict, handle)