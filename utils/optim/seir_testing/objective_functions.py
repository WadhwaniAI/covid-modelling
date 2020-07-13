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
from models.optim.seir_testing_dis import SEIR_Testing

def check(start_array, duration_array, choice_array, total_resource, days, less=False):
	for i in range(1,len(start_array)):
		if(start_array[i] < start_array[i-1] + duration_array[i-1]):
			return(0)
	if(start_array[-1]+duration_array[-1]>days):
		return(0)
	resource_spent = np.dot(duration_array, choice_array)
	if(not less):
		if(resource_spent != total_resource):
			return(0)
	else:
		if(resource_spent > total_resource):
			return(0)
	return(1)

# def get_impact(choice, factor=None):
# 	if(factor==None):
# 		return(1+(1.77/1.3)*choice)
# 	else:
# 		return(1+factor*choice)

def calculate_opt(intervention_day, intervention_duration, intervention_choice, days, params=None):

	assert(len(intervention_day) == len(intervention_duration))
	assert(len(intervention_duration) == len(intervention_choice))

	k = len(intervention_day)
	int_vec = np.zeros(days)
	for intvn in range(k):
		for i in range(intervention_day[intvn],min(intervention_day[intvn]+intervention_duration[intvn],days)):
			int_vec[i] = intervention_choice[intvn]

	
	solver = SEIR_Testing(int_vec=int_vec, **params)
	states_int_array = solver.solve_ode(time_step=1, total_no_of_days=days)
	
	S_coeficeint=0
	E_coeficeint=0
	I_coeficeint=1
	D_E_coeficeint=0
	D_I_coeficeint=0
	R_mild_coeficeint=0
	R_severe_home_coeficeint=0
	R_severe_hosp_coeficeint=0
	R_fatal_coeficeint=0
	C_coeficeint=0
	D_coeficeint=0
#   When we have joint optimization of time and qald
#     time_weight = 0.5 + (np.arange(days)/(2*days))
#     time_weight = time_weight[::-1]
#   when we have only qald
	time_weight = np.ones(days)
	coeficeint=[S_coeficeint,E_coeficeint,I_coeficeint,D_E_coeficeint,D_I_coeficeint,R_mild_coeficeint,R_severe_home_coeficeint,R_severe_hosp_coeficeint,\
				R_fatal_coeficeint,C_coeficeint,D_coeficeint]
	
	grad1 = np.dot(coeficeint, states_int_array)
	grad1 = np.dot(time_weight, grad1)
	return(grad1, states_int_array)

def calculate_opt_qald(intervention_day, intervention_duration, intervention_choice, days, params=None):
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days, params)
	return(grad1)

def calculate_opt_height(intervention_day, intervention_duration, intervention_choice, days, params=None):
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days, params)
	infection_array = states_int_array[2]
	height = np.max(infection_array)
	return(height)

def calculate_opt_time(intervention_day, intervention_duration, intervention_choice, days, params=None):
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days, params)
	infection_array = states_int_array[2]
	# maxy = int(np.max(infection_array)/0.01)
	# time = 0
	# for y in range(1,maxy+1):
	# 	x1 = np.argmax(infection_array>y*0.01)
	# 	x2 = len(infection_array)-np.argmax(infection_array[::-1]>y*0.01)-1
	# 	time = time + x1 + x2
	# time = time/(maxy*2)
	auci = np.sum(infection_array)
	running_sum = np.ones(len(infection_array))
	time_array = np.ones(10)
	for i in range(len(infection_array)):
		running_sum[i] = np.sum(infection_array[:i+1])
	for i in range(1,11):
		time_array[i-1] = np.argmax(running_sum>=auci*0.1*i)+1
	time = np.mean(time_array)
	# print(time_array)
	return(time)

def calculate_opt_burden(intervention_day, intervention_duration, intervention_choice, days, capacity=np.array([0.05]), params=None):
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days, params)
	infection_array = states_int_array[2]
	if(len(capacity)==1):
		burden = np.sum(infection_array[infection_array>=capacity[0]]-capacity[0])
	if(len(capacity)==2):
		capacity_array = np.ones_like(infection_array)
		capacity_array[0] = capacity[0]
		capacity_array[-1] = capacity[-1]
		for i in range(1,len(capacity_array)-1):
			capacity_array[i] = capacity_array[i-1]+((capacity[1]-capacity[0])/(len(capacity_array)-1))
		burden_array = infection_array-capacity_array
		burden_array = np.maximum(burden_array, np.zeros_like(burden_array))
		burden = np.sum(burden_array)
	return(burden)

def hp_calculate_opt_qald(variable_params, total_resource, days, params=None):
	intervention_day = np.array(variable_params['intervention_day'])
	intervention_duration = np.array(variable_params['intervention_duration'])
	intervention_choice = np.array(variable_params['intervention_choice'])
	
	if(not check(intervention_day, intervention_duration, intervention_choice, total_resource, days, less=True)):
		return(100)
	
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days, params)
		
	return(grad1)

def hp_calculate_opt_height(variable_params, total_resource, days, params=None):
	intervention_day = np.array(variable_params['intervention_day'])
	intervention_duration = np.array(variable_params['intervention_duration'])
	intervention_choice = np.array(variable_params['intervention_choice'])
	
	if(not check(intervention_day, intervention_duration, intervention_choice, total_resource, days, less=True)):
		return(100)
	
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days, params)
	infection_array = states_int_array[2]
	height = np.max(infection_array)
	return(height)

def hp_calculate_opt_time(variable_params, total_resource, days, params=None):
	intervention_day = np.array(variable_params['intervention_day'])
	intervention_duration = np.array(variable_params['intervention_duration'])
	intervention_choice = np.array(variable_params['intervention_choice'])
	
	if(not check(intervention_day, intervention_duration, intervention_choice, total_resource, days, less=True)):
		return(100)
	
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days, params)
	infection_array = states_int_array[2]
	# maxy = int(np.max(infection_array)/0.01)
	# time = 0
	# for y in range(1,maxy+1):
	# 	x1 = np.argmax(infection_array>y*0.01)
	# 	x2 = len(infection_array)-np.argmax(infection_array[::-1]>y*0.01)-1
	# 	time = time + x1 + x2
	# time = time/(maxy*2)
	auci = np.sum(infection_array)
	running_sum = np.ones(len(infection_array))
	time_array = np.ones(10)
	for i in range(len(infection_array)):
		running_sum[i] = np.sum(infection_array[:i+1])
	for i in range(1,11):
		time_array[i-1] = np.argmax(running_sum>=auci*0.1*i)+1
	time = np.mean(time_array)
	time = -1 * time
	return(time)

def hp_calculate_opt_burden(variable_params, total_resource, days, capacity=np.array([0.05]), params=None):
	intervention_day = np.array(variable_params['intervention_day'])
	intervention_duration = np.array(variable_params['intervention_duration'])
	intervention_choice = np.array(variable_params['intervention_choice'])
	
	if(not check(intervention_day, intervention_duration, intervention_choice, total_resource, days, less=True)):
		return(100)

	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days, params)
	infection_array = states_int_array[2]
	if(len(capacity)==1):
		burden = np.sum(infection_array[infection_array>=capacity[0]])
	if(len(capacity)==2):
		capacity_array = np.ones_like(infection_array)
		capacity_array[0] = capacity[0]
		capacity_array[-1] = capacity[-1]
		for i in range(1,len(capacity_array)-1):
			capacity_array[i] = capacity_array[i-1]+((capacity[1]-capacity[0])/(len(capacity_array)-1))
		burden_array = infection_array-capacity_array
		burden_array = np.maximum(burden_array, np.zeros_like(burden_array))
		burden = np.sum(burden_array)
	return(burden)

# def run_seir(int_vec, days):
# 	R0 = 3 
# 	T_treat = 30
# 	T_trans = T_treat/R0

# 	N = 1e5
# 	I0 = 100.0
	
# 	params = [T_trans, T_treat, N, int_vec]

# 	# S, E, I, R_mild, R_severe, R_severe_home, R_fatal, C, D
# 	state_init_values = [(N - I0)/N, I0/N, 0]
	
# 	solver = SIR(params, state_init_values)
# 	states_int_array = solver.solve_ode(time_step=1, total_no_of_days=days)
	
	
# 	S_coeficeint=0
# 	I_coeficeint=1
# 	R_coeficeint=0
# #   When we have joint optimization of time and qald
# #     time_weight = 0.5 + (np.arange(days)/(2*days))
# #     time_weight = time_weight[::-1]
# #   when we have only qald
# 	time_weight = np.ones(days)
	
# 	coeficeint=np.array([S_coeficeint,I_coeficeint,R_coeficeint])
	
# 	grad1 = np.dot(coeficeint, states_int_array)
# 	grad1 = np.dot(time_weight, grad1)
	
# 	return(grad1,states_int_array)

# def seir_qald(int_vec, days):
# 	grad1, states_int_array = run_seir(int_vec, days)
# 	return(grad1)

# def seir_height(int_vec, days):
# 	grad1, states_int_array = run_seir(int_vec, days)
# 	infection_array = states_int_array[1]
# 	height = np.max(infection_array)
# 	return(height)

# def seir_time(int_vec, days):
# 	grad1, states_int_array = run_seir(int_vec, days)
# 	infection_array = states_int_array[1]
# 	maxy = int(np.max(infection_array)/0.01)
# 	time = 0
# 	for y in range(1,maxy+1):
# 		x1 = np.argmax(infection_array>y*0.01)
# 		x2 = len(infection_array)-np.argmax(infection_array[::-1]>y*0.01)-1
# 		time = time + x1 + x2
# 	time = time/(maxy*2)
# 	return(time)

# def seir_burden(int_vec, days, capacity=np.array([0.1])):
# 	grad1, states_int_array = run_seir(int_vec, days)
# 	infection_array = states_int_array[1]
# 	if(len(capacity)==1):
# 		burden = np.sum(infection_array[infection_array>=capacity[0]])
# 	if(len(capacity)==2):
# 		capacity_array = np.ones_like(infection_array)
# 		capacity_array[0] = capacity[0]
# 		capacity_array[-1] = capacity[-1]
# 		for i in range(1,len(capacity_array)-1):
# 			capacity_array[i] = capacity_array[i-1]+((capacity[1]-capacity[0])/(len(capacity_array)-1))
# 		burden_array = infection_array-capacity_array
# 		burden_array = np.maximum(burden_array, np.zeros_like(burden_array))
# 		burden = np.sum(burden_array)
# 	return(burden)
