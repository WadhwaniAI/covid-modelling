import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
import datetime
import copy
import time
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
from viz import plot_forecast, plot_top_k_trials, plot_ptiles
from viz.fit import plot_histogram, plot_all_histograms, plot_mean_variance, plot_scatter, plot_kl_divergence, plot_heatmap_distribution_sigmas, plot_all_params, plot_all_losses, plot_all_buckets, plot_cv_in_params, plot_recovery_loss, plot_confidence_interval
import yaml
from data.dataloader import SimulatedDataLoader
from joblib import delayed, Parallel
import os


simulate_configs = {'seirhd':'seirhd_fixed.yaml'}
model_configs = {'seirhd':'default.yaml'}

params_to_fix = ['T_recov_fatal','T_inf']
out_file = 'losses_fixed_fatal_inf'
model_used = 'seirhd'
n_iters = 10
n_jobs = 10
n_trials = 6000
varying_perc = np.array([-0.1,-0.25,-0.5,-0.9])
# varying_perc = np.array([-0.1,-0.2,-0.25,-0.4,-0.5,-0.75,-0.9])

simulated_config_filename = simulate_configs[model_used]
with open(os.path.join("../../configs/simulated_data/", simulated_config_filename)) as configfile:
    simulated_config = yaml.load(configfile, Loader=yaml.SafeLoader)    
actual_params = simulated_config['params']

config_filename = model_configs[model_used]
config = read_config(config_filename)


def run_bo(config_params,param,test_value):
    conf = copy.deepcopy(config_params)
    if param in conf['variable_param_ranges']:
        del conf['variable_param_ranges'][param] 
    conf['default_params'][param] = test_value
    conf['fitting_method_params']['num_evals'] = n_trials
    predictions_dict = single_fitting_cycle(**conf)
    output = {}
    output['losses'] = predictions_dict['df_loss']
    output['best_params'] = predictions_dict['best_params']
    return output

varying_perc = np.concatenate([np.flipud(varying_perc),[0],-1*varying_perc])
required_params = actual_params.copy()
config_params = copy.deepcopy(config['fitting'])
del required_params['N']
for param in params_to_fix : 
    if param in config_params['variable_param_ranges']:
        del config_params['variable_param_ranges'][param]
    config_params['default_params'][param] = 'true' if 'hosp_ratio' in param else required_params[param]
    del required_params[param]

losses = {}

count = 0
for param,val in required_params.items():
    count += 1
    # if(count > 3):
        # break
    losses[param] = {}
    for perc_change in varying_perc : 
        print('param count : ',count,' perc_change: ',perc_change)
        losses[param][perc_change] = []
        if 'hosp_ratio' in param : 
            test_value = 'true'+str(perc_change)
        else : 
            test_value = val + val*perc_change
        scenario_losses = Parallel(n_jobs=n_jobs)(
            delayed(run_bo)(config_params,param,test_value) for _ in range(n_iters)
        )
        losses[param][perc_change] = scenario_losses

save_dir = '../../misc/predictions/sens_prof/'    
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(os.path.join(save_dir, out_file+".pickle"), 'wb') as handle:
    pkl.dump(losses, handle)


