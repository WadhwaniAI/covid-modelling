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
from viz.fit import plot_histogram, plot_all_histograms, plot_mean_variance, plot_scatter, plot_kl_divergence, plot_heatmap_distribution_sigmas, plot_all_params, plot_all_losses, plot_all_buckets
import yaml


# %%
predictions_dict = {}


# %%
output_folder = '../../misc/reports/{}'.format(datetime.datetime.now().strftime("%Y_%m%d_%H%M%S"))


# %%
predictions_dict.keys()


# %%
config_filenames = ['experiments/seirhd.yaml', 'experiments/seir_pu.yaml']
model_params = {
        'SEIRHD': [ 'lockdown_R0', 'T_inc', 'T_inf', 'T_inf', 'T_recov', 'T_recov_fatal', 'P_fatal', 'E_hosp_ratio', 'I_hosp_ratio'],
        'SEIR_PU': [ 'T_inc', 'T_inf_U', 'T_recov', 'T_recov_fatal', 'beta', 'd', 'P_fatal', 'I_hosp_ratio', 'E_hosp_ratio','Pu_pop_ratio'],
    }
model_names = list(model_params.keys())
configs = [read_config(config_filename) for config_filename in config_filenames]
# tuple format (state, district, starting_date, ending_date)
location_tuples = {
    # 'MUMBAI(Latest/14)' : ('Maharashtra', 'Mumbai', None, None, 2.0e+7,14),
    # 'MUMBAI(Latest/21)' : ('Maharashtra', 'Mumbai', None, None, 2.0e+7,21),
    # 'MUMBAI(Latest/28)' : ('Maharashtra', 'Mumbai', None, None, 2.0e+7,28),
    # 'MUMBAI(Latest/35)' : ('Maharashtra', 'Mumbai', None, None, 2.0e+7,35),
    'RANCHI(Latest/14)'  : ('Jharkhand', 'Ranchi', None, None, 0.14e+7),
    'RANCHI(Latest/21)'  : ('Jharkhand', 'Ranchi', None, None, 0.14e+7),
    'RANCHI(Latest/28)'  : ('Jharkhand', 'Ranchi', None, None, 0.14e+7),
    'RANCHI(Latest/35)'  : ('Jharkhand', 'Ranchi', None, None, 0.14e+7),
    # 'BOKARO'  : ('Jharkhand', 'Bokaro', None, None, 0.06e+7),
    # 'BOKARO'  : ('Jharkhand', 'Bokaro', None, None, 0.06e+7),
    # 'DHANBAD'  : ('Jharkhand', 'Dhanbad', None, None, 0.06e+7),
    # 'BANGALURU' : ('Karnataka', 'Bengaluru Urban', None, None),
    # 'ASSAM' : ('Assam', None, None, None),
    # 'JAIPUR' : ('Rajasthan', 'Jaipur', None, None)
}


# %%
num_rep_trials = 5
for tag, loc in location_tuples.items():
    predictions_dict[tag] = {}
    for j, config in enumerate(configs):
        predictions_dict[tag][model_names[j]] = {}
        config_params = copy.deepcopy(config['fitting'])
        config_params['data']['dataloading_params']['state'] = loc[0]
        config_params['data']['dataloading_params']['district'] = loc[1]
        config_params['split']['start_date'] = loc[2]
        config_params['split']['end_date'] = loc[3]
        config_params['default_params']['N'] = loc[4]
        if loc[1] != 'Mumbai':
            config_params['data']['smooth_jump'] = False
        for k in range(num_rep_trials):
            print(tag,config['fitting']['model'],k,config_params['default_params']['N'])
            predictions_dict[tag][model_names[j]][f'm{k}'] = single_fitting_cycle(**config_params) 


# %%
save_dir = '../../misc/predictions/train_period_ranchi/'    
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for tag, tag_dict in predictions_dict.items():
    with open(os.path.join(save_dir, tag + ".pickle"), 'wb') as handle:
        pkl.dump(tag_dict, handle)

# null.tpl [markdown]
# # ### Use the pickle file to read the predicitons_dict

# # %%
# with open('../../misc/predictions/predictions_dict.pickle', 'rb') as handle:
#     predictions_dict = pkl.load(handle)


# # %%
# wandb.init(project="covid-modelling")
# wandb.run.name = "degeneracy-exps-location"+wandb.run.name


# # %%
# plot_all_params(predictions_dict, model_params, method='ensemble_combined')


# # %%
# which_compartments = {model_names[i]: config['fitting']['loss']['loss_compartments'] for i, config in enumerate(configs)}
# plot_all_losses(predictions_dict, which_losses=['train', 'val'], which_compartments=which_compartments)


# # %%
# model_types = {'SEIRHD' : 'SEIRHD', 'SEIRHD Cons' : 'SEIRHD', 'SEIR_Undetected':'SEIR_Undetected','SEIR_PU':'SEIR_PU','SEIR_PU_Testing':'SEIR_PU_Testing'}
# plot_all_buckets(predictions_dict,  which_buckets=['S', 'I', 'E', 'I_U','I_D','P_U','R_severe','R_fatal','C','D'], compare='model',model_types=model_types)


# # %%



