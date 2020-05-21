import numpy as np
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
            T_trans = self.int_vec[t]*T_base
        except:
            T_trans = T_base

        # Init derivative vector
        dydt = np.zeros(len(y))
        # Write differential equations
        dydt[0] = -I*S/(T_trans)
        dydt[1] = I*S/(T_trans) - I/T_treat
        dydt[2] = I/T_treat


        return dydt

    def solve_ode(self, total_no_of_days=200, time_step=1):
        t_start = 0
        t_final = total_no_of_days
        time_steps = np.arange(t_start, total_no_of_days, time_step)

        S_array = np.ones(len(time_steps))
        I_array = np.ones(len(time_steps))
        R_array = np.ones(len(time_steps))

        [S,I,R] = self.state_init_values

        for i in range(len(time_steps)):
            S_array[i], I_array[i], R_array[i] = S, I, R
            t = time_steps[i]
            dydt = self.get_derivative(t,[S,I,R])
            S += dydt[0]
            I += dydt[1]
            R += dydt[2]
        sol = np.array([S_array, I_array, R_array])
        
        return sol
