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
from models.optim.sir_dis import SIR
from utils.optim.sir.objective_functions import *
from utils.optim.sir.optimization_methods import *

output_file = './test_conjecture.txt'
f = open(output_file, 'a')


days = 400
iterations = 20

acui_performance, height_performance, time_performance = [], [], []

for i in range(iterations):
	params = {}
	params['R0'] = 2+np.random.rand()
	params['T_treat'] = 14 + int(26*np.random.rand())
	f.write(params)
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
	f.write('AUCI (base): {}, Height (base): {}, Avg. Time (base): {}, Burden (base): {}, Peak Time (base): {}'.format(auci_base, height_base, time_base, burden_base, peak_time_base))

	



	# f.write('Optimizing AUCI')

	_, min_params = grid_search(num_int=1, days=days, objective='qald', sir_init=params)
	min_params = tpe_grid(num_int=1, days=days, min_params=min_params, objective='qald', iters=500, sir_init=params)
	auci_1 = calculate_opt_qald(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)
	_, min_params = grid_search(num_int=2, days=days, objective='qald', sir_init=params)
	min_params = tpe_grid(num_int=2, days=days, min_params=min_params, objective='qald', iters=1000, sir_init=params)
	auci_2 = calculate_opt_qald(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)
	_, min_params = grid_search(num_int=3, days=days, objective='qald', sir_init=params)
	min_params = tpe_grid(num_int=3, days=days, min_params=min_params, objective='qald', iters=1500, sir_init=params)
	auci_3 = calculate_opt_qald(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)

	f.write('best optimized AUCI : {}'.format(min(auci_1, auci_2, auci_3)))

	# f.write('Conjecture AUCI')
	conj_start_array = np.array([peak_time_base - 16])
	conj_duration_array = np.array([80])
	conj_choice_array = np.array([1])
	conj_auci = calculate_opt_qald(intervention_day=conj_start_array, intervention_duration=conj_duration_array,\
              intervention_choice=conj_choice_array, days=days, params=params)

	f.write('Conjecture Policy AUCI : {}'.format(conj_auci))
	performance = (auci_base - conj_auci) / (auci_base - min(auci_1, auci_2, auci_3))
	f.write('Performance of Conjecture: {}'.format(performance))
	acui_performance.append(performance)




	

	# f.write('Optimizing Height')

	_, min_params = grid_search(num_int=1, days=days, objective='height', sir_init=params)
	min_params = tpe_grid(num_int=1, days=days, min_params=min_params, objective='height', iters=500, sir_init=params)
	height_1 = calculate_opt_height(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)
	_, min_params = grid_search(num_int=2, days=days, objective='height', sir_init=params)
	min_params = tpe_grid(num_int=2, days=days, min_params=min_params, objective='height', iters=1000, sir_init=params)
	height_2 = calculate_opt_height(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)
	_, min_params = grid_search(num_int=3, days=days, objective='height', sir_init=params)
	min_params = tpe_grid(num_int=3, days=days, min_params=min_params, objective='height', iters=1500, sir_init=params)
	height_3 = calculate_opt_height(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)

	f.write('best optimized Height : {}'.format(min(height_1, height_2, height_3)))

	# f.write('Conjecture Height')
	conj_start_array = np.array([peak_time_base - 80])
	conj_duration_array = np.array([160])
	conj_choice_array = np.array([0.5])
	conj_height = calculate_opt_height(intervention_day=conj_start_array, intervention_duration=conj_duration_array,\
              intervention_choice=conj_choice_array, days=days, params=params)

	f.write('Conjecture Policy Height : {}'.format(conj_height))
	performance = (height_base - conj_height) / (height_base - min(height_1, height_2, height_3))
	f.write('Performance of Conjecture: {}'.format(performance))
	height_performance.append(performance)

	





	# f.write('Optimizing Avg. Time')

	_, min_params = grid_search(num_int=1, days=days, objective='time', sir_init=params)
	min_params = tpe_grid(num_int=1, days=days, min_params=min_params, objective='time', iters=500, sir_init=params)
	time_1 = calculate_opt_time(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)
	_, min_params = grid_search(num_int=2, days=days, objective='time', sir_init=params)
	min_params = tpe_grid(num_int=2, days=days, min_params=min_params, objective='time', iters=1000, sir_init=params)
	time_2 = calculate_opt_time(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)
	_, min_params = grid_search(num_int=3, days=days, objective='time', sir_init=params)
	min_params = tpe_grid(num_int=3, days=days, min_params=min_params, objective='time', iters=1500, sir_init=params)
	time_3 = calculate_opt_time(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
              intervention_choice=min_params['choice_array'], days=days, params=params)

	f.write('best optimized Avg. Time : {}'.format(max(time_1, time_2, time_3)))

	# f.write('Conjecture Avg. Time')
	conj_start_array = np.array([10])
	conj_duration_array = np.array([160])
	conj_choice_array = np.array([0.5])
	conj_time = calculate_opt_time(intervention_day=conj_start_array, intervention_duration=conj_duration_array,\
              intervention_choice=conj_choice_array, days=days, params=params)

	f.write('Conjecture Policy Avg. Time : {}'.format(conj_time))
	performance = (conj_time - time_base) / (max(time_1, time_2, time_3) - time_base)
	f.write('Performance of Conjecture: {}'.format(performance))
	time_performance.append(performance)




f.write('Performance of AUCI Conjecture: mean={}, std_deviation={}'.format(np.mean(np.array(acui_performance)),np.std(np.array(acui_performance))))
f.write('Performance of Height Conjecture: mean={}, std_deviation={}'.format(np.mean(np.array(height_performance)),np.std(np.array(height_performance))))
f.write('Performance of Avg Time Conjecture: mean={}, std_deviation={}'.format(np.mean(np.array(time_performance)),np.std(np.array(time_performance))))
f.close()

