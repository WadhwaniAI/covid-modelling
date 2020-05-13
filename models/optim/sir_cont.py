import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pdb

class SIR:
    def __init__(self, params, state_init_values):
        self.params = params
        self.state_init_values = state_init_values
        self.time_params = self.params[:-2]
        self.N = self.params[-2]
        self.int_vec = self.params[-1]


    def get_derivative(self, t, y):
        # Init state variables
        [S, I, R] = y

        # Init time parameters and probabilities
        T_base, T_treat = self.time_params
        
        # Modelling the intervention
        try:
            T_trans = self.int_vec[int(t)]*T_base
        except:
            T_trans = T_base

        # Init derivative vector
        dydt = np.zeros(y.shape)
        # Write differential equations
        dydt[0] = -I*S/(T_trans)
        dydt[1] = I*S/(T_trans) - I/T_treat
        dydt[2] = I/T_treat


        return dydt

    def solve_ode(self, total_no_of_days=200, time_step=2):
        t_start = 0
        t_final = total_no_of_days
        time_steps = np.arange(t_start, total_no_of_days + time_step, time_step)

        sol = solve_ivp(self.get_derivative, [t_start, t_final], 
                        self.state_init_values, method='RK45', t_eval=time_steps)
        
        return sol
