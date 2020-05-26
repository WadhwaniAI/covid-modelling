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
	return(1+2*choice)

def calculate_opt(intervention_day, intervention_duration, intervention_choice, days):
	R0 = 3 
	T_treat = 30
	T_trans = T_treat/R0

	N = 1e5
	I0 = 100.0

	assert(len(intervention_day) == len(intervention_duration))
	assert(len(intervention_duration) == len(intervention_choice))

	k = len(intervention_day)
	int_vec = np.ones(days)
	for intvn in range(k):
		for i in range(intervention_day[intvn],min(intervention_day[intvn]+intervention_duration[intvn],days)):
			int_vec[i] = get_impact(intervention_choice[intvn])
	
	params = [T_trans, T_treat, N, int_vec]

	state_init_values = [(N - I0)/N, I0/N, 0]
	
	solver = SIR(params, state_init_values)
	states_int_array = solver.solve_ode(time_step=1, total_no_of_days=days)
	
	S_coeficeint=0
	I_coeficeint=1
	R_coeficeint=0
#   When we have joint optimization of time and qald
#     time_weight = 0.5 + (np.arange(days)/(2*days))
#     time_weight = time_weight[::-1]
#   when we have only qald
	time_weight = np.ones(days)
	coeficeint=[S_coeficeint,I_coeficeint,R_coeficeint]
	
	grad1 = np.dot(coeficeint, states_int_array)
	grad1 = np.dot(time_weight, grad1)
	return(grad1, states_int_array)

def calculate_opt_qald(intervention_day, intervention_duration, intervention_choice, days):
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days)
	return(grad1)

def calculate_opt_height(intervention_day, intervention_duration, intervention_choice, days):
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days)
	infection_array = states_int_array[1]
	height = np.max(infection_array)
	return(height)

def calculate_opt_time(intervention_day, intervention_duration, intervention_choice, days):
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days)
	infection_array = states_int_array[1]
	time = np.argmax(infection_array)
	return(time)

def hp_calculate_opt_qald(variable_params, total_resource, days):
	intervention_day = np.array(variable_params['intervention_day'])
	intervention_duration = np.array(variable_params['intervention_duration'])
	intervention_choice = np.array(variable_params['intervention_choice'])
	
	if(not check(intervention_day, intervention_duration, intervention_choice, total_resource, days)):
		return(100)
	
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days)
		
	return(grad1)

def hp_calculate_opt_height(intervention_day, intervention_duration, intervention_choice, days):
	intervention_day = np.array(variable_params['intervention_day'])
	intervention_duration = np.array(variable_params['intervention_duration'])
	intervention_choice = np.array(variable_params['intervention_choice'])
	
	if(not check(intervention_day, intervention_duration, intervention_choice, total_resource, days)):
		return(100)
	
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days)
	infection_array = states_int_array[1]
	height = np.max(infection_array)
	return(height)

def hp_calculate_opt_time(intervention_day, intervention_duration, intervention_choice, days):
	intervention_day = np.array(variable_params['intervention_day'])
	intervention_duration = np.array(variable_params['intervention_duration'])
	intervention_choice = np.array(variable_params['intervention_choice'])
	
	if(not check(intervention_day, intervention_duration, intervention_choice, total_resource, days)):
		return(100)
	
	grad1, states_int_array = calculate_opt(intervention_day, intervention_duration, intervention_choice, days)
	infection_array = states_int_array[1]
	time = np.argmax(infection_array)
	return(time)

def run_seir(days, int_vec):
	R0 = 3 
	T_treat = 30
	T_trans = T_treat/R0

	N = 1e5
	I0 = 100.0
	
	params = [T_trans, T_treat, N, int_vec]

	# S, E, I, R_mild, R_severe, R_severe_home, R_fatal, C, D
	state_init_values = [(N - I0)/N, I0/N, 0]
	
	solver = SIR(params, state_init_values)
	states_int_array = solver.solve_ode(time_step=1, total_no_of_days=days)
	
	
	S_coeficeint=0
	I_coeficeint=1
	R_coeficeint=0
#   When we have joint optimization of time and qald
#     time_weight = 0.5 + (np.arange(days)/(2*days))
#     time_weight = time_weight[::-1]
#   when we have only qald
	time_weight = np.ones(days)
	
	coeficeint=np.array([S_coeficeint,I_coeficeint,R_coeficeint])
	
	grad1 = np.dot(coeficeint, states_int_array)
	grad1 = np.dot(time_weight, grad1)
	
	return(grad1,states_int_array)