import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import datetime
import pandas as pd
import itertools
import sys
import os
import math
sys.path.append('../..')
from joblib import Parallel, delayed
from functools import partial
from hyperopt import hp, tpe, fmin, Trials
from models.optim.sir_dis import SIR
from utils.optim.sir.objective_functions import *

def grid_search(num_int, objective='qald', total_resource=80, day0=10, days=500):
	min_val = 100
	min_params = {}
	params = []
	min_duration = 10
	max_duration = total_resource*np.array([4,2,1])
   
	if(num_int == 1):
		start_val = [start for start in range(day0, days-min_duration, 10)]
		choice_val = [0.25, 0.5, 1]
		duration_val = [duration for duration in range(10, 4*total_resource, 5)]
		start_tuple = list(itertools.product(start_val))
		choice_tuple = list(itertools.product(choice_val))
		duration_tuple = list(itertools.product(duration_val))
		inputs = list(itertools.product(start_tuple, duration_tuple, choice_tuple))
		for inp in inputs:
			start_array = np.array(inp[0])
			duration_array = np.array(inp[1])
			choice_array = np.array(inp[2])
			if(check(start_array, duration_array, choice_array, total_resource, days)):
				params.append([start_array, choice_array, duration_array])
		
	if(num_int == 2):
		start_val = [start for start in range(day0, days-min_duration, 30)]
		choice_val = [0.25, 0.5, 1]
		duration_val = [duration for duration in range(10, 4*total_resource, 10)]
		start_tuple = list(itertools.product(start_val, start_val))
		choice_tuple = list(itertools.product(choice_val, choice_val))
		duration_tuple = list(itertools.product(duration_val, duration_val))
		inputs = list(itertools.product(start_tuple, duration_tuple, choice_tuple))
		for inp in inputs:
			start_array = np.array(inp[0])
			duration_array = np.array(inp[1])
			choice_array = np.array(inp[2])
			if(check(start_array, duration_array, choice_array, total_resource, days)):
				params.append([start_array, choice_array, duration_array])
			
	if(num_int == 3):
		start_val = [start for start in range(day0, days-min_duration, 60)]
		choice_val = [0.25, 0.5, 1]
		duration_val = [duration for duration in range(10, 4*total_resource, 20)]
		start_tuple = list(itertools.product(start_val, start_val, start_val))
		choice_tuple = list(itertools.product(choice_val, choice_val, choice_val))
		duration_tuple = list(itertools.product(duration_val, duration_val, duration_val))
		inputs = list(itertools.product(start_tuple, duration_tuple, choice_tuple))
		for inp in inputs:
			start_array = np.array(inp[0])
			duration_array = np.array(inp[1])
			choice_array = np.array(inp[2])
			if(check(start_array, duration_array, choice_array, total_resource, days)):
				params.append([start_array, choice_array, duration_array])
		
							
	print(len(params))

	if(objective == 'qald'):
		value_array = Parallel(n_jobs=40)(delayed(calculate_opt_qald)(intervention_day=par[0], intervention_duration=par[2],\
																 intervention_choice=par[1], days = days) for par in params)
	if(objective == 'height'):
		value_array = Parallel(n_jobs=40)(delayed(calculate_opt_height)(intervention_day=par[0], intervention_duration=par[2],\
																 intervention_choice=par[1], days = days) for par in params)
	if(objective == 'time'):
		value_array = Parallel(n_jobs=40)(delayed(calculate_opt_time)(intervention_day=par[0], intervention_duration=par[2],\
																 intervention_choice=par[1], days = days) for par in params)
	value_array = np.array(value_array)
	min_val = np.min(value_array)
	i = np.argmin(value_array)
	min_params['start_array'] = params[i][0]
	min_params['duration_array'] = params[i][2]
	min_params['choice_array'] = params[i][1]
	
	return(min_val, min_params)


def tpe_opt(num_int, objective='qald', total_resource=80, day0=10, days=500):
	min_duration = 10
	if(num_int==1):
		start_val = [start for start in range(day0, days-min_duration, 10)]
		choice_val = [0.25, 0.5, 1]
		duration_val = [duration for duration in range(10, 4*total_resource+1, 5)]

		variable_params = {
			'intervention_day' : [hp.choice('intervention_day', start_val)],
			'intervention_duration' : [hp.choice('intervention_duration', duration_val)],
			'intervention_choice' : [hp.choice('intervention_choice', choice_val)],
		}
		
	if(num_int==2):
		start_val = [start for start in range(day0, days-min_duration, 10)]
		choice_val = [0.25, 0.5, 1]
		duration_val = [duration for duration in range(10, 4*total_resource+1, 10)]

		variable_params = {
			'intervention_day' : [hp.choice('id_0', start_val),hp.choice('id_1', start_val)],
			'intervention_duration' : [hp.choice('du_0', duration_val),hp.choice('du_1', duration_val)],
			'intervention_choice' : [hp.choice('ic_0', choice_val),hp.choice('ic_1', choice_val)],
		}

	if(objective == 'qald'):
		partial_calculate_opt = partial(hp_calculate_opt_qald, total_resource=total_resource, days=days)
	if(objective == 'height'):
		partial_calculate_opt = partial(hp_calculate_opt_height, total_resource=total_resource, days=days)
	if(objective == 'time'):
		partial_calculate_opt = partial(hp_calculate_opt_time, total_resource=total_resource, days=days) 
	
	searchspace = variable_params
	
	trials = Trials()
	best = fmin(partial_calculate_opt,
				space=searchspace,
				algo=tpe.suggest,
				max_evals=3000,
				trials=trials)
	
	return(best, trials)

def tpe_grid(num_int, min_params, objective='qald', total_resource=80, days=500):
	min_duration = 10
	window_start = 30
	if(num_int==1):
		start = int(min_params['start_array'][0] - window_start/2)
		end = int(min_params['start_array'][0] + window_start/2)
		start_val = [start for start in range(start, end)]
		choice_val = [min_params['choice_array'][0]]
		duration_val = [min_params['duration_array'][0]]

		variable_params = {
			'intervention_day' : [hp.choice('intervention_day', start_val)],
			'intervention_duration' : [hp.choice('intervention_duration', duration_val)],
			'intervention_choice' : [hp.choice('intervention_choice', choice_val)],
		}
		
	if(num_int==2):
		start = int(min_params['start_array'][0] - window_start/2)
		end = int(min_params['start_array'][0] + window_start/2)
		start_val_0 = [start for start in range(start, end)]
		start = int(min_params['start_array'][1] - window_start/2)
		end = int(min_params['start_array'][1] + window_start/2)
		start_val_1 = [start for start in range(start, end)]
		choice_val_0 = [min_params['choice_array'][0]]
		choice_val_1 = [min_params['choice_array'][1]]
		duration_val_0 = [min_params['duration_array'][0]]
		duration_val_1 = [min_params['duration_array'][1]]

		variable_params = {
			'intervention_day' : [hp.choice('id_0', start_val_0),hp.choice('id_1', start_val_1)],
			'intervention_duration' : [hp.choice('du_0', duration_val_0),hp.choice('du_1', duration_val_1)],
			'intervention_choice' : [hp.choice('ic_0', choice_val_0),hp.choice('ic_1', choice_val_1)],
		}

	if(objective == 'qald'):
		partial_calculate_opt = partial(hp_calculate_opt_qald, total_resource=total_resource, days=days)
	if(objective == 'height'):
		partial_calculate_opt = partial(hp_calculate_opt_height, total_resource=total_resource, days=days)
	if(objective == 'time'):
		partial_calculate_opt = partial(hp_calculate_opt_time, total_resource=total_resource, days=days) 

	searchspace = variable_params
	
	trials = Trials()
	best = fmin(partial_calculate_opt,
				space=searchspace,
				algo=tpe.suggest,
				max_evals=3000,
				trials=trials)
	
	return(best, trials)
