import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class SEIR_Testing():


    def __init__(self, vanilla_params, testing_params, state_init_values):
        
        for param_dict_name in ['vanilla_params', 'testing_params', 'state_init_values']:
            setattr(self, param_dict_name, eval(param_dict_name))


    def get_derivative(self, t, y):

        # Init state variables
        for i, _ in enumerate(y):
            y[i] = max(y[i], 0)
        S, E, I, D_E, D_I, R_mild, R_severe_home, R_severe_hosp, R_fatal, C, D = y

        # Init time parameters and probabilities
        for key in self.vanilla_params:
            setattr(self, key, self.vanilla_params[key])

        for key in self.testing_params:
            suffix = '_D' if key in self.vanilla_params else ''
            setattr(self, key + suffix, self.testing_params[key])

        # Modelling the intervention
        if t >= self.intervention_day:
            self.T_trans = self.intervention_amount * self.T_trans

        # Init derivative vector
        dydt = np.zeros(y.shape)
        
        theta_E = self.testing_rate_for_exposed
        psi_E = self.positive_test_rate_for_exposed
        theta_I = self.testing_rate_for_infected
        psi_I = self.positive_test_rate_for_infected

        # Write differential equations
        dydt[0] = - I * S / (self.T_trans) - (self.q / self.T_trans_D) * (S * D_I)
        dydt[1] = I * S / (self.T_trans) + (self.q / self.T_trans_D) * (S * D_I) - (E/ self.T_inc) - (theta_E * psi_E * E)
        dydt[2] = E / self.T_inc - I / self.T_inf - (theta_I * psi_I * I)
        dydt[3] = (theta_E * psi_E * E) - (1 / self.T_inc_D) * D_E
        dydt[4] = (theta_I * psi_I * I) + (1 / self.T_inc_D) * D_E - (1 / self.T_inf_D) * D_I
        dydt[5] = (1/self.T_inf)*(self.P_mild*I) + (1/self.T_inf_D)*(self.P_mild_D*D_I) - R_mild/self.T_recov_mild
        dydt[6] = (1/self.T_inf)*(self.P_severe*I) + (1/self.T_inf_D)*(self.P_severe_D*D_I) - R_severe_home/self.T_hosp 
        dydt[7] = R_severe_home/self.T_hosp - R_severe_hosp/self.T_recov_severe
        dydt[8] = (1/self.T_inf)*(self.P_fatal*I) + (1/self.T_inf_D)*(self.P_fatal_D*D_I) - R_fatal/self.T_death
        dydt[9] = R_mild/self.T_recov_mild + R_severe_hosp/self.T_recov_severe
        dydt[10] = R_fatal/self.T_death

        return dydt

    def solve_ode(self, total_no_of_days=200, time_step=2):
        t_start = 0
        t_final = total_no_of_days
        time_steps = np.arange(t_start, total_no_of_days + time_step, time_step)
        
        state_init_values_arr = [self.state_init_values[x] for x in self.state_init_values]

        sol = solve_ivp(self.get_derivative, [t_start, t_final], 
                        state_init_values_arr, method='RK45', t_eval=time_steps)

        return sol