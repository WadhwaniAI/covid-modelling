import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from collections import OrderedDict
import datetime

class SEIR_Testing():

    def __init__(self, pre_lockdown_R0=3, lockdown_R0=2.2, post_lockdown_R0=None, T_inf=2.9, T_inc=5.2, T_hosp=5, 
                 T_death=32, P_severe=0.2, P_fatal=0.02, T_recov_severe=14, T_recov_mild=11, N=7e6, init_infected=1,
                 intervention_day=10, q=0, testing_rate_for_exposed=0, positive_test_rate_for_exposed=1, 
                 testing_rate_for_infected=0, positive_test_rate_for_infected=1, intervention_removal_day=75, 
                 starting_date='2020-03-09', state_init_values=None, **kwargs):

        # If no value of post_lockdown R0 is provided, the model assumes the lockdown R0 post-lockdown
        if post_lockdown_R0 == None:
           post_lockdown_R0 = lockdown_R0

        P_mild = 1 - P_severe - P_fatal

        # define testing related parameters
        T_inf_detected = T_inf
        T_inc_detected = T_inc

        P_mild_detected = P_mild
        P_severe_detected = P_severe
        P_fatal_detected = P_fatal

        vanilla_params = {
            # R0 values
            'pre_lockdown_R0': pre_lockdown_R0, # R0 value pre-lockdown
            'lockdown_R0': lockdown_R0,  # R0 value during lockdown
            'post_lockdown_R0': post_lockdown_R0,  # R0 value post-lockdown

            # Transmission parameters
            'T_inc': T_inc,  # The incubation time of the infection
            'T_inf': T_inf,  # The duration for which an individual is infectious

            # Probability of contracting different types of infections
            'P_mild': P_mild,  # Probability of contracting a mild infection
            'P_severe': P_severe,  # Probability of contracting a severe infection
            'P_fatal': P_fatal,  # Probability of contracting a fatal infection

            # Clinical time parameters
            'T_recov_mild': T_recov_mild, # Time it takes for an individual with a mild infection to recover
            'T_recov_severe': T_recov_severe, # Time it takes for an individual with a severe infection to recover
            'T_hosp': T_hosp, # Time it takes for an individual to get hospitalised, after they have been diagnosed
            'T_death': T_death, #Time it takes for an individual with a fatal infection to die

            # Lockdown parameters
            'starting_date': starting_date,  # Datetime value that corresponds to Day 0 of modelling
            'intervention_day': intervention_day, # Number of days from the starting_date, after which lockdown is initiated
            'intervention_removal_day': intervention_removal_day, # Number of days from the starting_date, after which lockdown is removed
            'N': N
        }

        testing_params = {
            'T_inc': T_inc_detected,
            'T_inf': T_inf_detected,

            'P_mild': P_mild_detected,
            'P_severe': P_severe_detected,
            'P_fatal': P_fatal_detected,

            # Testing Parameters
            'q': q, # Perfection of quarantining : If q = 0, quarantining is perfect. q = 1. quarantining is absolutely imperfect
            'testing_rate_for_exposed': testing_rate_for_exposed, # Percentage of people in the Exposed bucket that are tested daily
            'sensitivity_rate_for_exposed': positive_test_rate_for_exposed, # Sensitivity of test that Exposed people undergo
            'testing_rate_for_infected': testing_rate_for_infected, # Percentage of people in the Infected bucket that are tested daily
            'sensitivity_rate_for_infected': positive_test_rate_for_infected # Sensitivity of test that Infected people undergo
        }

        if state_init_values == None:
            # S, E, D_E, D_I, I, R_mild, R_severe_home, R_severe_hosp, R_fatal, C, D
            state_init_values = OrderedDict()
            state_init_values['S'] = (N - init_infected)/N
            state_init_values['E'] = 0
            state_init_values['I'] = init_infected/N
            state_init_values['D_E'] = 0
            state_init_values['D_I'] = 0
            state_init_values['R_mild'] = 0
            state_init_values['R_severe_home'] = 0
            state_init_values['R_severe_hosp'] = 0
            state_init_values['R_fatal'] = 0
            state_init_values['C'] = 0
            state_init_values['D'] = 0

        for param_dict_name in ['vanilla_params', 'testing_params', 'state_init_values']:
            setattr(self, param_dict_name, eval(param_dict_name))

        # Init time parameters and probabilities
        for key in self.vanilla_params:
            setattr(self, key, self.vanilla_params[key])

        for key in self.testing_params:
            suffix = '_D' if key in self.vanilla_params else ''
            setattr(self, key + suffix, self.testing_params[key])


    def get_derivative(self, t, y):

        # Init state variables
        for i, _ in enumerate(y):
            y[i] = max(y[i], 0)
        S, E, I, D_E, D_I, R_mild, R_severe_home, R_severe_hosp, R_fatal, C, D = y

        # Modelling the behaviour post-lockdown
        if t >= self.intervention_removal_day:
            self.R0 = self.post_lockdown_R0
        # Modelling the behaviour lockdown
        elif t >= self.intervention_day:
            self.R0 = self.lockdown_R0
        # Modelling the behaviour pre-lockdown
        else:
            self.R0 = self.pre_lockdown_R0

        self.T_trans = self.T_inf/self.R0
        self.T_trans_D = self.T_inf_D/self.R0

        # Init derivative vector
        dydt = np.zeros(y.shape)
        
        self.theta_E = self.testing_rate_for_exposed
        self.psi_E = self.sensitivity_rate_for_exposed
        self.theta_I = self.testing_rate_for_infected
        self.psi_I = self.sensitivity_rate_for_infected

        # Write differential equations
        dydt[0] = - I * S / (self.T_trans) - (self.q / self.T_trans_D) * (S * D_I) # S
        dydt[1] = I * S / (self.T_trans) + (self.q / self.T_trans_D) * (S * D_I) - (E/ self.T_inc) - (self.theta_E * self.psi_E * E) # E
        dydt[2] = E / self.T_inc - I / self.T_inf - (self.theta_I * self.psi_I * I) # I
        dydt[3] = (self.theta_E * self.psi_E * E) - (1 / self.T_inc_D) * D_E # D_E
        dydt[4] = (self.theta_I * self.psi_I * I) + (1 / self.T_inc_D) * D_E - (1 / self.T_inf_D) * D_I # D_I 
        dydt[5] = (1/self.T_inf)*(self.P_mild*I) + (1/self.T_inf_D)*(self.P_mild_D*D_I) - R_mild/self.T_recov_mild # R_mild
        dydt[6] = (1/self.T_inf)*(self.P_severe*I) + (1/self.T_inf_D)*(self.P_severe_D*D_I) - R_severe_home/self.T_hosp # R_severe_home
        dydt[7] = R_severe_home/self.T_hosp - R_severe_hosp/self.T_recov_severe # R_severe_hosp
        dydt[8] = (1/self.T_inf)*(self.P_fatal*I) + (1/self.T_inf_D)*(self.P_fatal_D*D_I) - R_fatal/self.T_death # R_fatal
        dydt[9] = R_mild/self.T_recov_mild + R_severe_hosp/self.T_recov_severe # C
        dydt[10] = R_fatal/self.T_death # D

        return dydt

    def solve_ode(self, total_no_of_days=200, time_step=1, method='Radau'):
        t_start = 0
        t_final = total_no_of_days
        time_steps = np.arange(t_start, total_no_of_days + time_step, time_step)
        
        state_init_values_arr = [self.state_init_values[x] for x in self.state_init_values]

        sol = solve_ivp(self.get_derivative, [t_start, t_final], 
                        state_init_values_arr, method=method, t_eval=time_steps)

        return sol

    def return_predictions(self, sol):
        states_time_matrix = (sol.y*self.vanilla_params['N']).astype('int')
        dataframe_dict = {}
        for i, key in enumerate(self.state_init_values.keys()):
            dataframe_dict[key] = states_time_matrix[i]
        
        df_prediction = pd.DataFrame.from_dict(dataframe_dict)
        df_prediction['date'] = pd.date_range(self.starting_date, self.starting_date + datetime.timedelta(days=df_prediction.shape[0] - 1))
        columns = list(df_prediction.columns)
        columns.remove('date')
        df_prediction = df_prediction[['date'] + columns]

        df_prediction['hospitalised'] = df_prediction['R_severe_home'] + df_prediction['R_severe_hosp'] + df_prediction['R_fatal']
        df_prediction['recovered'] = df_prediction['C']
        df_prediction['deceased'] = df_prediction['D']
        df_prediction['infectious_unknown'] = df_prediction['I'] + df_prediction['D_I']
        df_prediction['total_infected'] = df_prediction['hospitalised'] + df_prediction['recovered'] + df_prediction['deceased']
        return df_prediction
