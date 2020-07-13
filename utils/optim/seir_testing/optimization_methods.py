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
from utils.optim.seir_testing.objective_functions import *
from pulp import *
from sys import getsizeof
import pdb


def launch_params(params, sir_init, days, capacity, objective):
	best_params = {}
	if(objective == 'qald'):
		value_array = Parallel(n_jobs=40)(delayed(calculate_opt_qald)(intervention_day=par[0], intervention_duration=par[2],\
																 intervention_choice=par[1], days = days, params=sir_init) for par in params)
	if(objective == 'height'):
		value_array = Parallel(n_jobs=40)(delayed(calculate_opt_height)(intervention_day=par[0], intervention_duration=par[2],\
																 intervention_choice=par[1], days = days, params=sir_init) for par in params)
	if(objective == 'time'):
		value_array = Parallel(n_jobs=40)(delayed(calculate_opt_time)(intervention_day=par[0], intervention_duration=par[2],\
																 intervention_choice=par[1], days = days, params=sir_init) for par in params)
	if(objective == 'burden'):
		value_array = Parallel(n_jobs=40)(delayed(calculate_opt_burden)(intervention_day=par[0], intervention_duration=par[2],\
																 intervention_choice=par[1], days = days, capacity=capacity, params=sir_init) for par in params)
	value_array = np.array(value_array)
	if(objective == 'time'):
		value_array = -1 * value_array
	best_val = np.min(value_array)
	i = np.argmin(value_array)
	best_params['start_array'] = params[i][0]
	best_params['duration_array'] = params[i][2]
	best_params['choice_array'] = params[i][1]
	return(best_val, best_params)


def grid_search(num_int, objective='qald', total_resource=80, day0=10, days=500, capacity=np.array([0.05]), sir_init=None):
	best_val = 100
	best_params = {}
	min_duration = 10
	total_resource = int(total_resource)
	max_duration = total_resource*np.array([4,2,1])
   
	if(num_int >= 1):
		params = []
		start_val = [start for start in range(day0, days-min_duration, 10)]
		choice_val = [0.25, 0.5, 1]
		duration_val = [duration for duration in range(10, 4*total_resource, 10)]
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
			if(len(params)>=10000):
				best_val_temp, best_params_temp = launch_params(params=params, sir_init=sir_init, days=days, capacity=capacity, objective=objective)
				if(best_val_temp<best_val):
					best_val = best_val_temp
					best_params = best_params_temp
				del params
				params = []


		best_val_temp, best_params_temp = launch_params(params=params, sir_init=sir_init, days=days, capacity=capacity, objective=objective)
		if(best_val_temp<best_val):
			best_val = best_val_temp
			best_params = best_params_temp

	if(num_int >= 2):
		params = []
		start_val = [start for start in range(day0, days-min_duration, 10)]
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
			if(len(params)>=10000):
				best_val_temp, best_params_temp = launch_params(params=params, sir_init=sir_init, days=days, capacity=capacity, objective=objective)
				if(best_val_temp<best_val):
					best_val = best_val_temp
					best_params = best_params_temp
				del params
				params = []
		best_val_temp, best_params_temp = launch_params(params=params, sir_init=sir_init, days=days, capacity=capacity, objective=objective)
		if(best_val_temp<best_val):
			best_val = best_val_temp
			best_params = best_params_temp
			
	if(num_int >= 3):
		params = []
		start_val = [start for start in range(day0, days-min_duration, 50)]
		choice_val = [0.25, 0.5, 1]
		duration_val = [duration for duration in range(10, 4*total_resource, 30)]
		start_tuple = list(itertools.product(start_val, start_val, start_val))
		choice_tuple = list(itertools.product(choice_val, choice_val, choice_val))
		duration_tuple = list(itertools.product(duration_val, duration_val, duration_val))
		# pdb.set_trace()
		# inputs = list(itertools.product(start_tuple, duration_tuple, choice_tuple))
		# for inp in inputs:
		for start in start_tuple:
			for duration in duration_tuple:
				for choice in choice_tuple:
					start_array = np.array(start)
					duration_array = np.array(duration)
					choice_array = np.array(choice)
					if(check(start_array, duration_array, choice_array, total_resource, days)):
						params.append([start_array, choice_array, duration_array])
					if(len(params)>=10000):
						best_val_temp, best_params_temp = launch_params(params=params, sir_init=sir_init, days=days, capacity=capacity, objective=objective)
						if(best_val_temp<best_val):
							best_val = best_val_temp
							best_params = best_params_temp
						del params
						params = []
		best_val_temp, best_params_temp = launch_params(params=params, sir_init=sir_init, days=days, capacity=capacity, objective=objective)
		if(best_val_temp<best_val):
			best_val = best_val_temp
			best_params = best_params_temp
		
							
	# print(len(params))

	# if(objective == 'qald'):
	# 	value_array = Parallel(n_jobs=40)(delayed(calculate_opt_qald)(intervention_day=par[0], intervention_duration=par[2],\
	# 															 intervention_choice=par[1], days = days, params=sir_init) for par in params)
	# if(objective == 'height'):
	# 	value_array = Parallel(n_jobs=40)(delayed(calculate_opt_height)(intervention_day=par[0], intervention_duration=par[2],\
	# 															 intervention_choice=par[1], days = days, params=sir_init) for par in params)
	# if(objective == 'time'):
	# 	value_array = Parallel(n_jobs=40)(delayed(calculate_opt_time)(intervention_day=par[0], intervention_duration=par[2],\
	# 															 intervention_choice=par[1], days = days, params=sir_init) for par in params)
	# if(objective == 'burden'):
	# 	value_array = Parallel(n_jobs=40)(delayed(calculate_opt_burden)(intervention_day=par[0], intervention_duration=par[2],\
	# 															 intervention_choice=par[1], days = days, capacity=capacity, params=sir_init) for par in params)
	# value_array = np.array(value_array)
	# if(objective == 'time'):
	# 	value_array = -1 * value_array
	# best_val = np.min(value_array)
	# i = np.argmin(value_array)
	# best_params['start_array'] = params[i][0]
	# best_params['duration_array'] = params[i][2]
	# best_params['choice_array'] = params[i][1]
	
	return(best_val, best_params)


def tpe_opt(num_int, objective='qald', total_resource=80, day0=10, days=500, capacity=np.array([0.05]), sir_init=None):
	min_duration = 10
	total_resource = int(total_resource)
	if(num_int==1):
		start_val_1 = [start for start in range(day0, days-min_duration, 10)]
		choice_val_1 = [0.25, 0.5, 1]
		duration_val_1 = [duration for duration in range(10, 4*total_resource+1, 5)]

		variable_params = {
			'intervention_day' : [hp.choice('id', start_val_1)],
			'intervention_duration' : [hp.choice('du', duration_val_1)],
			'intervention_choice' : [hp.choice('ic', choice_val_1)],
		}
		
	if(num_int==2):
		start_val_2 = [start for start in range(day0, days-min_duration, 20)]
		choice_val_2 = [0.25, 0.5, 1]
		duration_val_2 = [duration for duration in range(10, 4*total_resource+1, 10)]

		variable_params = {
			'intervention_day' : [hp.choice('id_0', start_val_2),hp.choice('id_1', start_val_2)],
			'intervention_duration' : [hp.choice('du_0', duration_val_2),hp.choice('du_1', duration_val_2)],
			'intervention_choice' : [hp.choice('ic_0', choice_val_2),hp.choice('ic_1', choice_val_2)],
		}

	if(num_int==3):
		start_val_3 = [start for start in range(day0, days-min_duration, 40)]
		choice_val_3 = [0.25, 0.5, 1]
		duration_val_3 = [duration for duration in range(10, 4*total_resource+1, 20)]

		variable_params = {
			'intervention_day' : [hp.choice('id_0', start_val_3),hp.choice('id_1', start_val_3),hp.choice('id_2', start_val_3)],
			'intervention_duration' : [hp.choice('du_0', duration_val_3),hp.choice('du_1', duration_val_3),hp.choice('du_2', duration_val_3)],
			'intervention_choice' : [hp.choice('ic_0', choice_val_3),hp.choice('ic_1', choice_val_3),hp.choice('ic_2', choice_val_3)],
		}

	if(objective == 'qald'):
		partial_calculate_opt = partial(hp_calculate_opt_qald, total_resource=total_resource, days=days, params=sir_init)
	if(objective == 'height'):
		partial_calculate_opt = partial(hp_calculate_opt_height, total_resource=total_resource, days=days, params=sir_init)
	if(objective == 'time'):
		partial_calculate_opt = partial(hp_calculate_opt_time, total_resource=total_resource, days=days, params=sir_init)
	if(objective == 'burden'):
		partial_calculate_opt = partial(hp_calculate_opt_burden, total_resource=total_resource, days=days, capacity=capacity, params=sir_init) 
	
	searchspace = variable_params
	
	trials = Trials()
	best = fmin(partial_calculate_opt,
				space=searchspace,
				algo=tpe.suggest,
				max_evals=3000,
				trials=trials)
	best_params = {}
	if(num_int==1):
		best_params['choice_array'] = np.array([choice_val_1[best['ic']]])
		best_params['start_array'] = np.array([start_val_1[best['id']]])
		best_params['duration_array'] = np.array([duration_val_1[best['du']]])
	if(num_int==2):
		best_params['choice_array'] = np.array([choice_val_2[best['ic_0']],choice_val_2[best['ic_1']]])
		best_params['start_array'] = np.array([start_val_2[best['id_0']],start_val_2[best['id_1']]])
		best_params['duration_array'] = np.array([duration_val_2[best['du_0']],duration_val_2[best['du_1']]])
	if(num_int==3):
		best_params['choice_array'] = np.array([choice_val_3[best['ic_0']],choice_val_3[best['ic_1']],choice_val_3[best['ic_2']]])
		best_params['start_array'] = np.array([start_val_3[best['id_0']],start_val_3[best['id_1']],start_val_3[best['id_2']]])
		best_params['duration_array'] = np.array([duration_val_3[best['du_0']],duration_val_3[best['du_1']],duration_val_3[best['du_2']]])

	return(best_params)

def tpe_grid(num_int, min_params, objective='qald', total_resource=80, day0=10, days=500, iters=3000, capacity=np.array([0.05]), sir_init=None):
	min_duration = 10
	window_start = 30
	window_duration = 30
	total_resource = int(total_resource)
	if(num_int==1):
		start = int(max(min_params['start_array'][0] - window_start/2, day0))
		end = int(min_params['start_array'][0] + window_start/2)
		start_val = [start for start in range(start, end)]
		choice_val = [min_params['choice_array'][0]]
		start = int(max(min_params['duration_array'][0] - window_duration/2, min_duration))
		end = int(min_params['duration_array'][0] + window_duration/2)
		duration_val = [duration for duration in range(start, end)]

		variable_params = {
			'intervention_day' : [hp.choice('id', start_val)],
			'intervention_duration' : [hp.choice('du', duration_val)],
			'intervention_choice' : [hp.choice('ic', choice_val)],
		}
		
	if(num_int==2):
		start = int(max(min_params['start_array'][0] - window_start/2, day0))
		end = int(min_params['start_array'][0] + window_start/2)
		start_val_0 = [start for start in range(start, end)]
		start = int(max(min_params['start_array'][1] - window_start/2, day0))
		end = int(min_params['start_array'][1] + window_start/2)
		start_val_1 = [start for start in range(start, end)]
		choice_val_0 = [min_params['choice_array'][0]]
		choice_val_1 = [min_params['choice_array'][1]]
		start = int(max(min_params['duration_array'][0] - window_duration/2, min_duration))
		end = int(min_params['duration_array'][0] + window_duration/2)
		duration_val_0 = [duration for duration in range(start, end)]
		start = int(max(min_params['duration_array'][1] - window_duration/2, min_duration))
		end = int(min_params['duration_array'][1] + window_duration/2)
		duration_val_1 = [duration for duration in range(start, end)]

		variable_params = {
			'intervention_day' : [hp.choice('id_0', start_val_0),hp.choice('id_1', start_val_1)],
			'intervention_duration' : [hp.choice('du_0', duration_val_0),hp.choice('du_1', duration_val_1)],
			'intervention_choice' : [hp.choice('ic_0', choice_val_0),hp.choice('ic_1', choice_val_1)],
		}

	if(num_int==3):
		start = int(max(min_params['start_array'][0] - window_start/2, day0))
		end = int(min_params['start_array'][0] + window_start/2)
		start_val_0 = [start for start in range(start, end)]
		start = int(max(min_params['start_array'][1] - window_start/2, day0))
		end = int(min_params['start_array'][1] + window_start/2)
		start_val_1 = [start for start in range(start, end)]
		start = int(max(min_params['start_array'][2] - window_start/2, day0))
		end = min(int(min_params['start_array'][2] + window_start/2),days)
		start_val_2 = [start for start in range(start, end)]
		choice_val_0 = [min_params['choice_array'][0]]
		choice_val_1 = [min_params['choice_array'][1]]
		choice_val_2 = [min_params['choice_array'][2]]
		start = int(max(min_params['duration_array'][0] - window_duration/2, min_duration))
		end = int(min_params['duration_array'][0] + window_duration/2)
		duration_val_0 = [duration for duration in range(start, end)]
		start = int(max(min_params['duration_array'][1] - window_duration/2, min_duration))
		end = int(min_params['duration_array'][1] + window_duration/2)
		duration_val_1 = [duration for duration in range(start, end)]
		start = int(max(min_params['duration_array'][2] - window_duration/2, min_duration))
		end = int(min_params['duration_array'][2] + window_duration/2)
		duration_val_2 = [duration for duration in range(start, end)]

		variable_params = {
			'intervention_day' : [hp.choice('id_0', start_val_0),hp.choice('id_1', start_val_1),hp.choice('id_2', start_val_2)],
			'intervention_duration' : [hp.choice('du_0', duration_val_0),hp.choice('du_1', duration_val_1),hp.choice('du_2', duration_val_2)],
			'intervention_choice' : [hp.choice('ic_0', choice_val_0),hp.choice('ic_1', choice_val_1),hp.choice('ic_2', choice_val_2)],
		}

	if(objective == 'qald'):
		partial_calculate_opt = partial(hp_calculate_opt_qald, total_resource=total_resource, days=days, params=sir_init)
	if(objective == 'height'):
		partial_calculate_opt = partial(hp_calculate_opt_height, total_resource=total_resource, days=days, params=sir_init)
	if(objective == 'time'):
		partial_calculate_opt = partial(hp_calculate_opt_time, total_resource=total_resource, days=days, params=sir_init)
	if(objective == 'burden'):
		partial_calculate_opt = partial(hp_calculate_opt_burden, total_resource=total_resource, days=days, capacity=capacity, params=sir_init) 

	searchspace = variable_params
	
	trials = Trials()
	best = fmin(partial_calculate_opt,
				space=searchspace,
				algo=tpe.suggest,
				max_evals=iters,
				trials=trials)
	best_params = {}
	if(num_int==1):
		best_params['choice_array'] = np.array([choice_val[best['ic']]])
		best_params['start_array'] = np.array([start_val[best['id']]])
		best_params['duration_array'] = np.array([duration_val[best['du']]])
	if(num_int==2):
		best_params['choice_array'] = np.array([choice_val_0[best['ic_0']],choice_val_1[best['ic_1']]])
		best_params['start_array'] = np.array([start_val_0[best['id_0']],start_val_1[best['id_1']]])
		best_params['duration_array'] = np.array([duration_val_0[best['du_0']],duration_val_1[best['du_1']]])
	if(num_int==3):
		best_params['choice_array'] = np.array([choice_val_0[best['ic_0']],choice_val_1[best['ic_1']],choice_val_2[best['ic_2']]])
		best_params['start_array'] = np.array([start_val_0[best['id_0']],start_val_1[best['id_1']],start_val_2[best['id_2']]])
		best_params['duration_array'] = np.array([duration_val_0[best['du_0']],duration_val_1[best['du_1']],duration_val_2[best['du_2']]])

	return(best_params)

# def get_grad(int_vec, start, objective_function):
#     days = len(int_vec)
#     num_pts = 3  # number of points used for derievative.
#     grad_vec = np.ones(days)
#     for i in range(start, days):
#     #     i represents the ith dimension of T_transi
#         val = int_vec[i]
#         window_size = 0.005
#         values = np.random.uniform(val-window_size, val+window_size, num_pts)
#         f_values = np.ones_like(values)
#         for j,value in enumerate(values):
#             int_vec[i] = value
#             f_values[j] = objective_function(int_vec,days)
#         values = np.append(values,val)
#         int_vec[i] = val
#         fval = objective_function(int_vec,days)
#         f_values = np.append(f_values,fval)
#         derivative = np.gradient(f_values, values, edge_order=2)
#         grad_vec[i] = derivative[-1]
#     return(grad_vec)

# def opt_step(grad, days, start, total_resource):
#     prob = LpProblem("Minimization over constrained space",LpMinimize)

#     day_list = list((np.arange(days-start)).astype(str))

#     int_days = LpVariable.dicts("day",day_list,lowBound=1,upBound=3,cat='Continuous')

# #     grad = get_grad(np.ones(400))

#     prob += lpSum([grad[int(i)]*int_days[i] for i in day_list]) #objective function
#     prob += lpSum([int_days[f] for f in day_list]) <= float(days+2*total_resource-start) #TotalResource
#     prob.solve()
#     opt_val = np.ones(days)
#     for v in prob.variables():
#         idx = v.name.split('_')[-1]
#         opt_val[int(idx)+start] = v.varValue
        
#     return(opt_val)

# def frank_wolfe(days, start=10, total_resource=80, objective='qald'):
# 	num_iter = 30
# 	best_val = 50
# 	int_vec = np.ones(days)
# 	best_int_vec = np.ones(days)
# 	if(objective=='qald'):
# 		objective_function = eval('seir_qald')
# 	if(objective=='height'):
# 		objective_function = eval('seir_height')
# 	if(objective=='time'):
# 		objective_function = eval('seir_time')
# 	if(objective=='burden'):
# 		objective_function = eval('seir_burden')
# 	for i in range(num_iter):
# 	    gradient = get_grad(int_vec,start,objective_function)
# 	    opt = opt_step(gradient,days,start,total_resource)
# 	    gamma = 2.0/(3+i)
# 	    int_vec = int_vec + gamma*(opt-int_vec)
# 	    value = objective_function(int_vec=int_vec,days=days)
# 	    if(objective=='time'):
# 	    	value = -1 * value
# 	    if(value < best_val):
# 	        best_val = value
# 	        best_int_vec = int_vec
# 	        print(best_val)
# 	return(best_int_vec)