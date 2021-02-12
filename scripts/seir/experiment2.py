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

simulate_configs = {'seirhd':'seirhd_fixed.yaml','seirhd_beta':'seirhd_beta_fixed.yaml'}
model_configs = {'seirhd':'default.yaml','seirhd_beta':'experiments/seirhd_beta.yaml'}

out_file = 'exp2_trial1'
model_used = 'seirhd_beta'
n_iters = 10
n_jobs = 10
n_trials = 3000
# progress_filename = "./progress/" + out_file + ".txt"
# log_file = open(progress_filename, 'wb')

format_str = '%d-%m-%Y' # The format

start_date = datetime.datetime.strptime('1-1-2020',format_str).date()
val_periods = [14,28,42,56,70,84,98,112]
# train_periods = [14,28,42,56,70,84,98,112,126,140]
train_periods = [28,70]
# end_dates = ['1-8-2020','1-12-2020']
end_dates = ['1-8-2020','1-10-2020','1-11-2020','31-12-2020']

config_filename = model_configs[model_used]
config = read_config(config_filename)
config_params = copy.deepcopy(config['fitting'])
config_params['fitting_method_params']['num_evals'] = n_trials
config_params['data']['add_noise'] = True
config_params['data']['dataloading_params']['generate'] = False
config_params['data']['dataloading_params']['filename'] = '../../data/data/simulated_data/seirhd_beta.csv'

def run_bo(config_params, run_tuple):
    train,val,end_date = run_tuple['train'],run_tuple['val'],run_tuple['end_date']
    print (str(train) + " " + str(val) + " " + str(end_date) + "\n")
    conf = copy.deepcopy(config_params)
    conf['split']['end_date'] = end_date
    conf['split']['train_period'] = train
    conf['split']['val_period'] = val
    predictions_dict = single_fitting_cycle(**conf)
    output = {}
    output['prediction_dict'] = predictions_dict
    output['run_tuple'] = run_tuple
    return output

run_tuple = []
for val in val_periods : 
    for train in train_periods : 
        for end_date in end_dates : 
            end_date = datetime.datetime.strptime(end_date,format_str).date()
            if(end_date - datetime.timedelta(days=train+val)) < start_date :
                continue
            for i in range(n_iters):
                scenario = {'val':val,'train':train,'end_date':end_date,'iter':i}
                run_tuple.append(scenario)
print(run_tuple)
# exit(1)

predictions_dicts = Parallel(n_jobs=n_jobs)(
    delayed(run_bo)(config_params, run_tuple[i]) for i in range(len(run_tuple))
)


output_dict =  {}
output_dict['predictions_dicts'] = predictions_dicts
output_dict['val_periods'] = val_periods
output_dict['train_periods'] = train_periods
output_dict['end_dates'] = end_dates
output_dict['model_config'] = config_params

save_dir = '../../misc/predictions/experiment2/'    
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(os.path.join(save_dir, out_file+".pickle"), 'wb') as handle:
    pkl.dump(output_dict, handle)

# log_file.close()
