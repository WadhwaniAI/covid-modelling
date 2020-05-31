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
from models.optim.seirgoh_dis import SEIR

def check(start_array, duration_array, choice_array, total_resource, days):
	for i in range(1,len(start_array)):
		if(start_array[i] <= start_array[i-1] + duration_array[i-1]):
			return(0)
	if(start_array[-1]+duration_array[-1]>=days):
		return(0)
	resource_spent = np.dot(duration_array, choice_array)
	if(resource_spent != total_resource):
		return(0)
	else:
		return(1)

def get_impact(choice):
	return(1+1*choice)

def calculate_opt(intervention_day, intervention_duration, intervention_choice, days, params=None):
	if(params==None):
		R0 = 3 
		T_inf = 30
		T_inc = 5
		T_death = 50
		T_recov_mild = (50 - T_inf)
		T_hosp = 10
		T_recov_severe = (50 - T_inf)		

	else:
		R0 = params['R0']
		T_inf = params['T_inf']
		T_inc = params['T_inc']
		T_death = params['T_death']
		T_recov_mild = params['T_recov_mild']
		T_hosp = params['T_hosp']
		T_recov_severe = params['T_recov_severe']

	P_severe = 0.3
	P_fatal = 0.04
	P_mild = 1 - P_severe - P_fatal
	T_trans = T_inf/R0
	N = 1e5
	I0 = 100.0

	assert(len(intervention_day) == len(intervention_duration))
	assert(len(intervention_duration) == len(intervention_choice))

	k = len(intervention_day)
	int_vec = np.ones(days)
	for intvn in range(k):
		for i in range(intervention_day[intvn],min(intervention_day[intvn]+intervention_duration[intvn],days)):
			int_vec[i] = get_impact(intervention_choice[intvn])
	
	params = [T_trans, T_inc, T_inf, T_recov_mild, T_hosp, T_recov_severe, T_death, 
              P_mild, P_severe, P_fatal, N, int_vec]

	state_init_values = [(N - I0)/N, 0, I0/N, 0, 0, 0, 0, 0, 0]
	
	solver = SEIR(params, state_init_values)
	states_int_array = solver.solve_ode(time_step=1, total_no_of_days=days)
	
	S_coeficeint=0
    E_coeficeint=0
    I_coeficeint=0.7
    R_mild_coeficeint=0.7
    R_severe_coeficeint=0.9
    R_severe_hosp_coeficeint=0.9
    R_R_fatal_coeficeint=0.9
    C_coeficeint=0
    D_coeficeint=1
#   When we have joint optimization of time and qald
#     time_weight = 0.5 + (np.arange(days)/(2*days))
#     time_weight = time_weight[::-1]
#   when we have only qald
	time_weight = np.ones(days)
	coeficeint=[S_coeficeint,E_coeficeint,I_coeficeint,R_mild_coeficeint,R_severe_coeficeint,R_severe_hosp_coeficeint,\
                R_R_fatal_coeficeint,C_coeficeint,D_coeficeint]
	
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
	maxy = int(np.max(infection_array)/0.01)
	time = 0
	for y in range(1,maxy+1):
		x1 = np.argmax(infection_array>y*0.01)
		x2 = len(infection_array)-np.argmax(infection_array[::-1]>y*0.01)-1
		time = time + x1 + x2
	time = time/(maxy*2)
	return(time)

def calculate_opt_burden(intervention_day, intervention_duration, intervention_choice, days, capacity=np.array([0.1]), params=None):
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

def hp_calculate_opt_qald(variable_params, total_resource, days, params=None):
	intervention_day = np.array(variable_params['intervention_day'])
	intervention_duration = np.array(variable_params['intervention_duration'])
	intervention_choice = np.array(variable_params['intervention_choice'])
	
	if(not check(intervention_day, intervention_duration, intervention_choice, total_resource, days)):
		return(100)
	
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days, params)
		
	return(grad1)

def hp_calculate_opt_height(variable_params, total_resource, days, params=None):
	intervention_day = np.array(variable_params['intervention_day'])
	intervention_duration = np.array(variable_params['intervention_duration'])
	intervention_choice = np.array(variable_params['intervention_choice'])
	
	if(not check(intervention_day, intervention_duration, intervention_choice, total_resource, days)):
		return(100)
	
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days, params)
	infection_array = states_int_array[2]
	height = np.max(infection_array)
	return(height)

def hp_calculate_opt_time(variable_params, total_resource, days, params=None):
	intervention_day = np.array(variable_params['intervention_day'])
	intervention_duration = np.array(variable_params['intervention_duration'])
	intervention_choice = np.array(variable_params['intervention_choice'])
	
	if(not check(intervention_day, intervention_duration, intervention_choice, total_resource, days)):
		return(100)
	
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days, params)
	infection_array = states_int_array[2]
	maxy = int(np.max(infection_array)/0.01)
	time = 0
	for y in range(1,maxy+1):
		x1 = np.argmax(infection_array>y*0.01)
		x2 = len(infection_array)-np.argmax(infection_array[::-1]>y*0.01)-1
		time = time + x1 + x2
	time = time/(maxy*2)
	time = -1 * time
	return(time)

def hp_calculate_opt_burden(variable_params, total_resource, days, capacity, params=None):
	intervention_day = np.array(variable_params['intervention_day'])
	intervention_duration = np.array(variable_params['intervention_duration'])
	intervention_choice = np.array(variable_params['intervention_choice'])
	
	if(not check(intervention_day, intervention_duration, intervention_choice, total_resource, days)):
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

def run_seir(int_vec, days, params=None):
	if(params==None):
		R0 = 3 
		T_inf = 30
		T_inc = 5
		T_death = 50
		T_recov_mild = (50 - T_inf)
		T_hosp = 10
		T_recov_severe = (50 - T_inf)		

	else:
		R0 = params['R0']
		T_inf = params['T_inf']
		T_inc = params['T_inc']
		T_death = params['T_death']
		T_recov_mild = params['T_recov_mild']
		T_hosp = params['T_hosp']
		T_recov_severe = params['T_recov_severe']

	P_severe = 0.3
	P_fatal = 0.04
	P_mild = 1 - P_severe - P_fatal
	T_trans = T_inf/R0
	N = 1e5
	I0 = 100.0
	
	params = [T_trans, T_inc, T_inf, T_recov_mild, T_hosp, T_recov_severe, T_death, 
              P_mild, P_severe, P_fatal, N, int_vec]

	state_init_values = [(N - I0)/N, 0, I0/N, 0, 0, 0, 0, 0, 0]

	# S, E, I, R_mild, R_severe, R_severe_home, R_fatal, C, D
	
	solver = SIR(params, state_init_values)
	states_int_array = solver.solve_ode(time_step=1, total_no_of_days=days)
	
	
	S_coeficeint=0
    E_coeficeint=0
    I_coeficeint=0.7
    R_mild_coeficeint=0.7
    R_severe_coeficeint=0.9
    R_severe_hosp_coeficeint=0.9
    R_R_fatal_coeficeint=0.9
    C_coeficeint=0
    D_coeficeint=1
#   When we have joint optimization of time and qald
#     time_weight = 0.5 + (np.arange(days)/(2*days))
#     time_weight = time_weight[::-1]
#   when we have only qald
	time_weight = np.ones(days)
	
	coeficeint=[S_coeficeint,E_coeficeint,I_coeficeint,R_mild_coeficeint,R_severe_coeficeint,R_severe_hosp_coeficeint,\
                R_R_fatal_coeficeint,C_coeficeint,D_coeficeint]
	
	grad1 = np.dot(coeficeint, states_int_array)
	grad1 = np.dot(time_weight, grad1)
	
	return(grad1,states_int_array)

def seir_qald(int_vec, days):
	grad1, states_int_array = run_seir(int_vec, days)
	return(grad1)

def seir_height(int_vec, days):
	grad1, states_int_array = run_seir(int_vec, days)
	infection_array = states_int_array[2]
	height = np.max(infection_array)
	return(height)

def seir_time(int_vec, days):
	grad1, states_int_array = run_seir(int_vec, days)
	infection_array = states_int_array[2]
	maxy = int(np.max(infection_array)/0.01)
	time = 0
	for y in range(1,maxy+1):
		x1 = np.argmax(infection_array>y*0.01)
		x2 = len(infection_array)-np.argmax(infection_array[::-1]>y*0.01)-1
		time = time + x1 + x2
	time = time/(maxy*2)
	return(time)

def seir_burden(int_vec, days, capacity=np.array([0.1])):
	grad1, states_int_array = run_seir(int_vec, days)
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
