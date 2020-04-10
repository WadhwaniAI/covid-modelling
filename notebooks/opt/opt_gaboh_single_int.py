import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import datetime
import pandas as pd
import sys
import os
import math
sys.path.append('..')
os.chdir("..")
import scipy.optimize as opt

from models.opt_gaboh_single_int import SEIR

def calculate_opt(intervention):
    R0 = 2.2 
    T_inf = 2.9
    T_trans = T_inf/R0
    T_inc = 5.2
    T_recov_mild = (14 - T_inf)
    T_hosp = 5
    T_recov_severe = (31.5 - T_inf)
    T_death = 32

    P_severe = 0.2
    P_fatal = 0.02
    P_mild = 1 - P_severe - P_fatal

    N = 7e6
    I0 = 1.0

    # new_R0 = 0.74
    # intervention_amount = R0/new_R0
    intervention_day = intervention[0]
    intervention_duration = intervention[1]
    intervention_amount = intervention[2]
 
    
    params = [T_trans, T_inc, T_inf, T_recov_mild, T_hosp, T_recov_severe, T_death, 
              P_mild, P_severe, P_fatal, N, intervention_day, intervention_amount, intervention_duration]

    # S, E, I, R_mild, R_severe, R_severe_home, R_fatal, C, D
    state_init_values = [(N - I0)/N, 0, I0/N, 0, 0, 0, 0, 0, 0]
    
    solver = SEIR(params, state_init_values)
    sol = solver.solve_ode(time_step=1, total_no_of_days=400)
    states_int_array = (sol.y*N).astype('int')
    
    E = states_int_array[1]
    I = states_int_array[2]
    H = states_int_array[5]
    F = states_int_array[8]
    
    # S_coeficeint=1
    # E_coeficeint=0.825
    # I_coeficeint=0.75
    # R_mild_coeficeint=0.625
    # R_severe_coeficeint=0.5
    # R_severe_home_coeficeint=0.325
    # R_R_fatal_coeficeint=0.25
    # C_coeficeint=1
    # D_coeficeint=0
    
    S_coeficeint=1
    E_coeficeint=0
    I_coeficeint=0
    R_mild_coeficeint=0
    R_severe_coeficeint=0
    R_severe_home_coeficeint=0
    R_R_fatal_coeficeint=0
    C_coeficeint=1
    D_coeficeint=0
    # plt.plot(range(len(states_int_array[0])),states_int_array[0])
    # plt.plot(range(len(states_int_array[0])),states_int_array[2])
    # plt.plot(range(len(states_int_array[0])),states_int_array[7])
    # plt.plot(range(len(states_int_array[0])),states_int_array[8])
    
    coeficeint=[S_coeficeint,E_coeficeint,I_coeficeint,R_mild_coeficeint,R_severe_coeficeint,R_severe_home_coeficeint,R_severe_home_coeficeint,R_R_fatal_coeficeint,C_coeficeint,D_coeficeint]
    
    # objective = N*intervention_duration*0.8 + 1e5 + F[-1]*1e4 + np.sum(H)
    objective=0
    for i in range(8):
        objective+=-coeficeint[i]*np.sum(states_int_array[i]/1e8)
        # print(np.sum(states_int_array[i]/1e8))
    return(objective)
    
# print(calculate_opt([50, 100, 1.5]))

# res = opt.minimize(calculate_opt, [50,100,1], method='nelder-mead',options={'xatol': 1e0, 'disp': True, 'fatol': 1e-2, 'maxiter': 1e4, 'maxfev': 1e4})
# print(res)
# res = opt.minimize(calculate_opt, [50,100,3], method='Powell',options={'xtol': 1e0, 'disp': True, 'ftol': 1e-2, 'maxiter': 1e4, 'maxfev': 1e4})

# obj = np.zeros(300)
# for i in range(300):
#     obj[i] = calculate_opt([100, i, 3])
# ind = np.arange(300)   # the x locations for the groups
# width = 0.95        # the width of the bars: can also be len(x) sequence
# plt.figure(figsize=(12, 12))
# plt.plot(ind, obj, label='COST')
# plt.ylabel('optimization objective')
# plt.xlabel('intervention duration')
# plt.legend()
# plt.show()

i1=55
i2=100
i3=1.2
r=10

obj_S = np.zeros((2*r,2*r))
for i in range(i1-r,i1+r):
    for j in range(i2-r,i2+r):
        obj_S[i-(i1-r),j-(i2-r)]=-calculate_opt([i, j, i3])
ax = sns.heatmap(obj_S)