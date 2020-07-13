import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
# import datetime
# import pandas as pd
import itertools
import sys
import os
import math
sys.path.append('../..')
# from joblib import Parallel, delayed
# from functools import partial
# from hyperopt import hp, tpe, fmin, Trials
from models.optim.seir_testing_dis import SEIR_Testing
from utils.optim.seir_testing.objective_functions import *
from utils.optim.seir_testing.optimization_methods import *
# from matplotlib import colors

# ---------------------------------- Initializing Parameters ----------------------------------

params_dict = {}
params_dict['T_inf'] = 20
params_dict['T_inc'] = 4
params_dict['P_severe'] = 0.3
params_dict['P_fatal'] = 0.0
params_dict['T_recov_severe'] = 10
params_dict['factor'] = 1

# params_dict['I_hosp_ratio'] = 0.39
# params_dict['E_hosp_ratio'] = 0.5

# self.T_inf=3
# # self.T_inf=4
# self.T_inc=4
# # self.T_inc=5
# self.P_severe=0.3
# # self.P_severe=0.99
# self.P_fatal=0
# # self.P_fatal=0.3
# self.T_recov_severe=10
# # self.T_recov_severe=60
# self.factor=1
# # self.factor=2

# ---------------------------------- Baseline No Intervention values --------------------------------


days = 400
params = params_dict
grad1, states_base = calculate_opt(intervention_day=np.array([100]), intervention_duration=np.array([50]), intervention_choice=np.array([0]), days=days, params=params)
height = calculate_opt_height(intervention_day=np.array([100]), intervention_duration=np.array([50]), intervention_choice=np.array([0]), days=days, params=params)
time = calculate_opt_time(intervention_day=np.array([100]), intervention_duration=np.array([50]), intervention_choice=np.array([0]), days=days, params=params)
burden = calculate_opt_burden(intervention_day=np.array([100]), intervention_duration=np.array([50]), intervention_choice=np.array([0]), days=days, capacity=np.array([0.05]), params=params)
print('No Intervention')
print('AUCI: {}'.format(grad1))
print('Height: {}'.format(height))
print('Time: {}'.format(time))
print('Burden: {}'.format(burden))

# ind = np.arange(days)   # the x locations for the groups
# plt.figure(figsize=(8,8))
# plt.plot(ind, states_base[2], label='without intervention')
# plt.ylabel('infected fraction')
# plt.xlabel('days')
# plt.legend()
# plt.show()


# -------------------------------- Doing baseline computations ----------------------------------------


def get_stats(min_params, params=None):
	val, states = calculate_opt(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
			  intervention_choice=min_params['choice_array'], days=days, params=params)
	height = calculate_opt_height(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
				  intervention_choice=min_params['choice_array'], days=days, params=params)
	time = calculate_opt_time(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
				  intervention_choice=min_params['choice_array'], days=days, params=params)
	burden = calculate_opt_burden(intervention_day=min_params['start_array'], intervention_duration=min_params['duration_array'],\
				  intervention_choice=min_params['choice_array'], days=days, params=params)
	# print(val, height, time, burden)
	print('AUCI: {}'.format(val))
	print('Height: {}'.format(height))
	print('Time: {}'.format(time))
	print('Burden: {}'.format(burden))
	return(states)

# def get_plot(states, fig, num, min_params=None):
#     ind = np.arange(len(states[2]))
#     ax = fig.add_subplot(1, 1, 1)
#     if(not num):
#         ax.plot(ind, states[2], label='without intervention', linestyle='dashed')
#     else:
#         k = len(min_params['start_array'])
#         for i in range(k):
#             xi = ind[min_params['start_array'][i]:min_params['start_array'][i]+min_params['duration_array'][i]]
#             yi = states[2][min_params['start_array'][i]:min_params['start_array'][i]+min_params['duration_array'][i]]
#             choice = min_params['choice_array'][i]
#             ax.plot(xi, yi, label='strength = '+str(choice)+', '+str(num)+' intervention', linewidth=3)
#         ax.plot(ind, states[2], alpha=0.5, label='strength = 0, '+str(num)+' intervention')
#     ax.set_ylabel('infected people')
#     ax.set_xlabel('days')
#     ax.set_title('Grid_Bayesian: Burden')
#     ax.legend()
#     return(fig)


objective_list = ['qald', 'time', 'height', 'burden']
for obj in objective_list:
	print('Optimizing {}'.format(obj))
	print('num_int = 1')
	params=params_dict
	_, min_params = grid_search(num_int=1, days=days, objective=obj, sir_init=params, total_resource=0.2*days)
	# min_params = tpe_grid(num_int=1, days=days, min_params=min_params, objective='height', iters=2000, sir_init=params, total_resource=0.2*days)
	print('Action Vector:\n{}'.format(min_params))
	states = get_stats(min_params, params)

	# fig = plt.figure(figsize=(8, 8))
	# fig = get_plot(fig=fig,states=states_base,num=0)
	# fig = get_plot(fig=fig,states=states,num=1,min_params=min_params)

	print('num_int = 2')
	_, min_params = grid_search(num_int=2, days=days, objective=obj, sir_init=params, total_resource=0.2*days)
	# min_params = tpe_grid(num_int=2, days=days, min_params=min_params, objective='height', iters=4000, sir_init=params, total_resource=0.2*days)
	print('Action Vector:\n{}'.format(min_params))
	states = get_stats(min_params, params=params)
	# fig = get_plot(fig=fig,states=states,num=2,min_params=min_params)

	# print('num_int = 3')
	# _, min_params = grid_search(num_int=3, days=days, objective=obj, sir_init=params, total_resource=0.2*days)
	# min_params = tpe_grid(num_int=3, days=days, min_params=min_params, objective='height', iters=6000, sir_init=params, total_resource=0.2*days)
	# print('Action Vector:\n{}'.format(min_params))
	# states = get_stats(min_params, params=params)
	# fig = get_plot(fig=fig,states=states,num=3,min_params=min_params)

