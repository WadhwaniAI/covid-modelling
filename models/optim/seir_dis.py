import numpy as np
import pdb

class SEIR:
    def __init__(self, params, state_init_values):
        self.params = params
        self.state_init_values = state_init_values
        self.time_params = self.params[:-5]
        self.p_params = self.params[-5:-2]
        self.N = self.params[-2]
        self.int_vec = self.params[-1]


    def get_derivative(self, t, y):
        # Init state variables
        [S, E, I, R_mild, R_severe, R_severe_hosp, R_fatal, C, D] = y

        # Init time parameters and probabilities
        T_base, T_inc, T_inf, T_recov_mild, T_hosp, T_recov_severe, T_death = self.time_params
        P_mild, P_severe, P_fatal = self.p_params
        

        try:
            T_trans = self.int_vec[t]*T_base
        except:
            T_trans = T_base

        # Init derivative vector
        dydt = np.zeros(len(y))
        # Write differential equations
        dydt[0] = -I*S/(T_trans)
        dydt[1] = I*S/(T_trans) - E/T_inc
        dydt[2] = E/T_inc - I/T_inf
        dydt[3] = (P_mild*I)/T_inf - R_mild/T_recov_mild
        dydt[4] = (P_severe*I)/T_inf - R_severe/T_hosp
        dydt[5] = R_severe/T_hosp - R_severe_hosp/T_recov_severe
        dydt[6] = (P_fatal*I)/T_inf - R_fatal/T_death
        dydt[7] = R_mild/T_recov_mild + R_severe_hosp/T_recov_severe
        dydt[8] = R_fatal/T_death

        return dydt

    def solve_ode(self, total_no_of_days=200, time_step=1):
        t_start = 0
        t_final = total_no_of_days
        time_steps = np.arange(t_start, total_no_of_days, time_step)

        S_array = np.ones(len(time_steps))
        E_array = np.ones(len(time_steps))
        I_array = np.ones(len(time_steps))
        R_mild_array = np.ones(len(time_steps))
        R_severe_array = np.ones(len(time_steps))
        R_severe_hosp_array = np.ones(len(time_steps))
        R_fatal_array = np.ones(len(time_steps))
        C_array = np.ones(len(time_steps))
        D_array = np.ones(len(time_steps))


        [S, E, I, R_mild, R_severe, R_severe_hosp, R_fatal, C, D] = self.state_init_values

        for i in range(len(time_steps)):
            S_array[i], E_array[i], I_array[i], R_mild_array[i], R_severe_array[i], R_severe_hosp_array[i], R_fatal_array[i], C_array[i], D_array[i] \
                                                                                            = S, E, I, R_mild, R_severe, R_severe_hosp, R_fatal, C, D
            t = time_steps[i]
            dydt = self.get_derivative(t,[S, E, I, R_mild, R_severe, R_severe_hosp, R_fatal, C, D])
            S += dydt[0]
            E += dydt[1]
            I += dydt[2]
            R_mild += dydt[3]
            R_severe += dydt[4]
            R_severe_hosp += dydt[5]
            R_fatal += dydt[6]
            C += dydt[7]
            D += dydt[8]
        sol = np.array([S_array, E_array, I_array, R_mild_array, R_severe_array, R_severe_hosp_array, R_fatal_array,C_array, D_array])
        
        return sol
