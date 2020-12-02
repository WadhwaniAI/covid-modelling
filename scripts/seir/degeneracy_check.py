# %%
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
from viz import plot_forecast, plot_top_k_trials, plot_ptiles
from viz.fit import plot_histogram, plot_all_histograms, plot_mean_variance, plot_scatter, plot_kl_divergence, plot_heatmap_distribution_sigmas, save_result_summary, plot_all_params, plot_all_losses, plot_all_buckets, plot_cv_in_params, plot_recovery_loss
import yaml
from data.dataloader import SimulatedDataLoader


# %%
predictions_dict = {}


# %%
output_folder = '../../misc/reports/{}'.format(datetime.datetime.now().strftime("%Y_%m%d_%H%M%S"))


# %%
predictions_dict.keys()

# %%
config_filenames = ['experiments/seir_pu.yaml']
model_params = {
        'SEIR_PU': [ 'T_inc', 'T_inf_U', 'T_recov', 'T_recov_fatal', 'beta', 'd', 'P_fatal', 'I_hosp_ratio', 'E_hosp_ratio','Pu_pop_ratio']
    }
model_types = {'SEIRHD' : 'SEIRHD', 'SEIRHD Free' : 'SEIRHD', 'SEIRHD Cons' : 'SEIRHD', 'SEIR_Undetected':'SEIR_Undetected','SEIR_PU':'SEIR_PU','SEIR_PU_Testing':'SEIR_PU_Testing'}
model_names = list(model_params.keys())
configs = [read_config(config_filename) for config_filename in config_filenames]

param_tuples = {
    # 'all':{'I_hosp_ratio': 0.5, 'E_hosp_ratio': 0.5,'Pu_pop_ratio': 0.3,'P_fatal':0.3,'d':0.2,'beta':0.25,'T_recov_fatal':25,'T_recov':15,'T_inf_U':10,'T_inc':4.5},
    # 'fixed_params':{'I_hosp_ratio': 'true', 'E_hosp_ratio': 'true','Pu_pop_ratio': 'true'},
    # 'lat_time':{'I_hosp_ratio': 'true', 'E_hosp_ratio': 'true','Pu_pop_ratio': 'true','T_recov_fatal':25,'T_recov':15,'T_inf_U':10,'T_inc':4.5},
    # 'ratios':{'I_hosp_ratio': 'true', 'E_hosp_ratio': 'true','Pu_pop_ratio': 'true','P_fatal':0.3,'d':0.2},

    'fixed_params':{'I_hosp_ratio': 'true', 'E_hosp_ratio': 'true'},
    # 'lat_time':{'I_hosp_ratio': 'true', 'E_hosp_ratio': 'true','T_inf_U':10,'d':0.2},
    'ratios':{'I_hosp_ratio': 'true', 'E_hosp_ratio': 'true','T_recov_fatal':25,'T_recov':15},
}


# %%
configs = [read_config(config_filename) for config_filename in config_filenames]
num_rep_trials = 4
for tag, loc in param_tuples.items():
    predictions_dict[tag] = {}
    for j, config in enumerate(configs):
        predictions_dict[tag][model_names[j]] = {}
        config_params = copy.deepcopy(config['fitting'])
        for param in loc:
            if param in config_params['variable_param_ranges']:
                del config_params['variable_param_ranges'][param] 
            config_params['default_params'][param] = loc[param]
        print ('variable param ranges:', config_params['variable_param_ranges'])
        print ('default param ranges:', config_params['default_params'])
        for k in range(num_rep_trials):
            predictions_dict[tag][model_names[j]][f'm{k}'] = single_fitting_cycle(**config_params) 


# %%
save_dir = '../../misc/predictions/deg_exp_odin/'    
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for tag, tag_dict in predictions_dict.items():
    with open(os.path.join(save_dir, tag + ".pickle"), 'wb') as handle:
        pkl.dump(tag_dict, handle)
# ### Use the pickle file to read the predicitons_dict