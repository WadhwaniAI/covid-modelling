import numpy as np
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
from utils.optim.seir_testing.optimization_methods import *


days = 300
iterations = 40
np.random.seed(0)

acui_performance, height_performance, time_performance = [], [], []

for i in range(iterations):
	output_file = './seir_testing_rules_performance.txt'
	f = open(output_file, 'a')
	
	# Sampling parameters
	
	params = {}
	params['T_inf'] = 3 + np.random.rand()
	params['T_inc'] = 4 + np.random.rand()
	params['P_severe'] = 0.3 + np.random.rand()*0.69
	params['P_fatal'] = np.random.rand()*0.3
	params['I_hosp_ratio'] = 0.3 + np.random.rand()*0.2
	params['E_hosp_ratio'] = 0.1 + np.random.rand()*0.8
	params['T_recov_severe'] = 10 + np.random.rand()*50
	params['factor'] = 1 + np.random.rand()
	f.write(str(params)+'\n')

	# calculating the base values

	auci_base = calculate_opt_qald(intervention_day=np.array([100]), intervention_duration=np.array([50]), intervention_choice=np.array([0]),\
									 days=days, params=params)
	height_base = calculate_opt_height(intervention_day=np.array([100]), intervention_duration=np.array([50]), intervention_choice=np.array([0]),\
										 days=days, params=params)
	time_base = calculate_opt_time(intervention_day=np.array([100]), intervention_duration=np.array([50]), intervention_choice=np.array([0]),\
								 days=days, params=params)
	burden_base = calculate_opt_burden(intervention_day=np.array([100]), intervention_duration=np.array([50]), intervention_choice=np.array([0]),\
					days=days, capacity=np.array([0.05]), params=params)
	_, states = calculate_opt(intervention_day=np.array([100]), intervention_duration=np.array([50]), intervention_choice=np.array([0]), days=days, params=params)
	peak_time_base = np.argmax(states[1])
	f.write('AUCI (base): {}, Height (base): {}, Avg. Time (base): {}, Burden (base): {}, Peak Time (base): {}'.format(auci_base, height_base, time_base, burden_base, peak_time_base) + '\n')



	# f.write('Optimizing AUCI')

	_, min_params = grid_search(num_int=1, days=days, objective='qald', sir_init=params, total_resource=0.2*days)
	min_params = tpe_grid(num_int=1, days=days, min_params=min_params, objective='qald', iters=500, sir_init=params, total_resource=0.2*days)
	auci_1 = calculate_opt_qald(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)
	_, min_params = grid_search(num_int=2, days=days, objective='qald', sir_init=params, total_resource=0.2*days)
	min_params = tpe_grid(num_int=2, days=days, min_params=min_params, objective='qald', iters=1000, sir_init=params, total_resource=0.2*days)
	auci_2 = calculate_opt_qald(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)
	_, min_params = grid_search(num_int=3, days=days, objective='qald', sir_init=params, total_resource=0.2*days)
	min_params = tpe_grid(num_int=3, days=days, min_params=min_params, objective='qald', iters=1500, sir_init=params, total_resource=0.2*days)
	auci_3 = calculate_opt_qald(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)

	f.write('best optimized AUCI : {}'.format(min(auci_1, auci_2, auci_3)) + '\n')

	# f.write('Rule AUCI')
	conj_start_array = np.array([peak_time_base - 12])
	conj_duration_array = np.array([60])
	conj_choice_array = np.array([1])
	conj_auci = calculate_opt_qald(intervention_day=conj_start_array, intervention_duration=conj_duration_array,\
              intervention_choice=conj_choice_array, days=days, params=params)

	f.write('Rule Policy AUCI : {}'.format(conj_auci) + '\n')
	performance = (auci_base - conj_auci) / (auci_base - min(auci_1, auci_2, auci_3))
	f.write('Performance of Rule: {}'.format(performance) + '\n')
	acui_performance.append(performance)


	

	# f.write('Optimizing Height')

	_, min_params = grid_search(num_int=1, days=days, objective='height', sir_init=params, total_resource=0.2*days)
	min_params = tpe_grid(num_int=1, days=days, min_params=min_params, objective='height', iters=500, sir_init=params, total_resource=0.2*days)
	height_1 = calculate_opt_height(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)
	_, min_params = grid_search(num_int=2, days=days, objective='height', sir_init=params, total_resource=0.2*days)
	min_params = tpe_grid(num_int=2, days=days, min_params=min_params, objective='height', iters=1000, sir_init=params, total_resource=0.2*days)
	height_2 = calculate_opt_height(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)
	_, min_params = grid_search(num_int=3, days=days, objective='height', sir_init=params, total_resource=0.2*days)
	min_params = tpe_grid(num_int=3, days=days, min_params=min_params, objective='height', iters=1500, sir_init=params, total_resource=0.2*days)
	height_3 = calculate_opt_height(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)

	f.write('best optimized Height : {}'.format(min(height_1, height_2, height_3)) + '\n')

	# f.write('Rule Height')
	conj_start_array = np.array([peak_time_base - 20])
	conj_duration_array = np.array([60])
	conj_choice_array = np.array([1])
	conj_height = calculate_opt_height(intervention_day=conj_start_array, intervention_duration=conj_duration_array,\
              intervention_choice=conj_choice_array, days=days, params=params)

	f.write('Rule Policy Height : {}'.format(conj_height) + '\n')
	performance = (height_base - conj_height) / (height_base - min(height_1, height_2, height_3))
	f.write('Performance of Rule: {}'.format(performance) + '\n')
	height_performance.append(performance)

	



	# f.write('Optimizing Avg. Time')

	_, min_params = grid_search(num_int=1, days=days, objective='time', sir_init=params, total_resource=0.2*days)
	min_params = tpe_grid(num_int=1, days=days, min_params=min_params, objective='time', iters=500, sir_init=params, total_resource=0.2*days)
	time_1 = calculate_opt_time(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)
	_, min_params = grid_search(num_int=2, days=days, objective='time', sir_init=params, total_resource=0.2*days)
	min_params = tpe_grid(num_int=2, days=days, min_params=min_params, objective='time', iters=1000, sir_init=params, total_resource=0.2*days)
	time_2 = calculate_opt_time(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)
	_, min_params = grid_search(num_int=3, days=days, objective='time', sir_init=params, total_resource=0.2*days)
	min_params = tpe_grid(num_int=3, days=days, min_params=min_params, objective='time', iters=1500, sir_init=params, total_resource=0.2*days)
	time_3 = calculate_opt_time(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)

	f.write('best optimized Avg. Time : {}'.format(max(time_1, time_2, time_3)) + '\n')

	# f.write('Rule Avg. Time')
	conj_start_array = np.array([10])
	conj_duration_array = np.array([60])
	conj_choice_array = np.array([1])
	conj_time = calculate_opt_time(intervention_day=conj_start_array, intervention_duration=conj_duration_array,\
              intervention_choice=conj_choice_array, days=days, params=params)

	f.write('Rule Policy Avg. Time : {}'.format(conj_time) + '\n')
	performance = (conj_time - time_base) / (max(time_1, time_2, time_3) - time_base)
	f.write('Performance of Rule: {}'.format(performance) + '\n')
	time_performance.append(performance)

	f.close()



output_file = './seir_testing_rules_performance.txt'
f = open(output_file, 'a')

f.write('Performance values of AUCI Rule: {}'.format(acui_performance) + '\n')
f.write('Performance values of Height Rule: {}'.format(height_performance) + '\n')
f.write('Performance values of Avg Time Rule: {}'.format(time_performance) + '\n')

f.write('Performance of AUCI Rule: mean={}, std_deviation={}'.format(np.mean(np.array(acui_performance)),np.std(np.array(acui_performance))) + '\n')
f.write('Performance of Height Rule: mean={}, std_deviation={}'.format(np.mean(np.array(height_performance)),np.std(np.array(height_performance))) + '\n')
f.write('Performance of Avg Time Rule: mean={}, std_deviation={}'.format(np.mean(np.array(time_performance)),np.std(np.array(time_performance))) + '\n')

f.close()


