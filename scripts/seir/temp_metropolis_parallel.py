#!/usr/bin/env python
# coding: utf-8

# In[1]:


import wandb
import yaml
from viz import plot_forecast, plot_top_k_trials, plot_ptiles
from utils.generic.logging import log_wandb
from utils.fitting.loss import Loss_Calculator
from utils.generic.enums import Columns
from utils.generic.config import read_config, make_date_key_str
from utils.generic.create_report import save_dict_and_create_report
from main.seir.sensitivity import calculate_sensitivity_and_plot
from main.seir.forecast import get_forecast, forecast_all_trials, create_all_trials_csv, create_decile_csv_new
from main.seir.fitting import single_fitting_cycle
import models
from data.processing import get_data
import sys
import time
import copy
import datetime
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


sys.path.append('../../')


# In[3]:


predictions_dict = {}


# In[4]:


config_filename = 'uncer.yaml'
config = read_config(config_filename)

wandb_config = read_config(config_filename, preprocess=False)
wandb_config = make_date_key_str(wandb_config)


# In[5]:


output_folder = '../../misc/reports/{}'.format(
    datetime.datetime.now().strftime("%Y_%m%d_%H%M%S"))


# ## Perform M1 and M2 fits

# In[7]:


predictions_dict['m1'] = single_fitting_cycle(
    **copy.deepcopy(config['fitting']))

m2_params = copy.deepcopy(config['fitting'])
m2_params['split']['val_period'] = 0
predictions_dict['m2'] = single_fitting_cycle(**m2_params)

predictions_dict['fitting_date'] = datetime.datetime.now().strftime("%Y-%m-%d")


# In[ ]:


predictions_dict['m1']['best_params']


# In[ ]:


predictions_dict['m2']['best_params']


# ## Loss Dataframes

# ### M1 Loss DataFrame

# In[ ]:


predictions_dict['m1']['df_loss']


# ### M2 Loss DataFrame

# In[ ]:


predictions_dict['m2']['df_loss']


# ## Sensitivity Plot

# In[ ]:


predictions_dict['m1']['plots']['sensitivity'], _, _ = calculate_sensitivity_and_plot(
    predictions_dict, config, which_fit='m1')
predictions_dict['m2']['plots']['sensitivity'], _, _ = calculate_sensitivity_and_plot(
    predictions_dict, config, which_fit='m2')


# ## Plot Forecasts

# In[ ]:


predictions_dict['m2']['forecasts'] = {}
predictions_dict['m2']['forecasts']['best'] = get_forecast(predictions_dict, train_fit='m2',
                                                           model=config['fitting']['model'],
                                                           days=config['forecast']['forecast_days'])

predictions_dict['m2']['plots']['forecast_best'] = plot_forecast(predictions_dict,
                                                                 'test',
                                                                 error_bars=True)

predictions_dict['m1']['trials_processed'] = forecast_all_trials(predictions_dict, train_fit='m1',
                                                                 model=config['fitting']['model'],
                                                                 forecast_days=config['forecast']['forecast_days'])

predictions_dict['m2']['trials_processed'] = forecast_all_trials(predictions_dict, train_fit='m2',
                                                                 model=config['fitting']['model'],
                                                                 forecast_days=config['forecast']['forecast_days'])

kforecasts = plot_top_k_trials(predictions_dict, train_fit='m2',
                               k=config['forecast']['num_trials_to_plot'],
                               which_compartments=config['forecast']['plot_topk_trials_for_columns'])

predictions_dict['m2']['plots']['forecasts_topk'] = {}
for column in config['forecast']['plot_topk_trials_for_columns']:
    predictions_dict['m2']['plots']['forecasts_topk'][column.name] = kforecasts[column]


# ## Uncertainty + Uncertainty Forecasts

# In[ ]:


uncertainty_args = {'predictions_dict': predictions_dict,
                    **config['uncertainty']['uncertainty_params']}
uncertainty = config['uncertainty']['method'](**uncertainty_args)


# In[ ]:


uncertainty.beta_loss


# In[ ]:


uncertainty_forecasts = uncertainty.get_forecasts()
for key in uncertainty_forecasts.keys():
    predictions_dict['m2']['forecasts'][key] = uncertainty_forecasts[key]['df_prediction']

predictions_dict['m2']['forecasts']['ensemble_mean'] = uncertainty.ensemble_mean_forecast


# In[ ]:


predictions_dict['m2']['beta'] = uncertainty.beta
predictions_dict['m2']['beta_loss'] = uncertainty.beta_loss
predictions_dict['m2']['deciles'] = uncertainty_forecasts


# In[ ]:


predictions_dict['m2']['plots']['forecast_best_50'] = plot_forecast(predictions_dict,
                                                                    (config['fitting']['data']['dataloading_params']['state'],
                                                                     config['fitting']['data']['dataloading_params']['district']),
                                                                    fits_to_plot=['best', 50], error_bars=False)
predictions_dict['m2']['plots']['forecast_best_80'] = plot_forecast(predictions_dict,
                                                                    (config['fitting']['data']['dataloading_params']['state'],
                                                                     config['fitting']['data']['dataloading_params']['district']),
                                                                    fits_to_plot=['best', 80], error_bars=False)
predictions_dict['m2']['plots']['forecast_ensemble_mean_50'] = plot_forecast(predictions_dict,
                                                                             (config['fitting']['data']['dataloading_params']['state'],
                                                                              config['fitting']['data']['dataloading_params']['district']),
                                                                             fits_to_plot=['ensemble_mean', 50], error_bars=False)


# In[ ]:


ptiles_plots = plot_ptiles(
    predictions_dict, which_compartments=config['forecast']['plot_ptiles_for_columns'])
predictions_dict['m2']['plots']['forecasts_ptiles'] = {}
for column in config['forecast']['plot_ptiles_for_columns']:
    predictions_dict['m2']['plots']['forecasts_ptiles'][column.name] = ptiles_plots[column]


# ## Create Report

# In[ ]:


save_dict_and_create_report(
    predictions_dict, config, ROOT_DIR=output_folder, config_filename=config_filename)


# ## Create Output CSV

# In[ ]:


df_output = create_decile_csv_new(predictions_dict)
df_output.to_csv(f'{output_folder}/deciles.csv')


# ## Log on W&B

# In[ ]:


wandb.init(project="covid-modelling", config=wandb_config)


# In[ ]:


log_wandb(predictions_dict)


# ## Create All Trials Output

# In[ ]:


df_all = create_all_trials_csv(predictions_dict)
df_all.to_csv(f'{output_folder}/all_trials.csv')


# In[ ]:
