import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from collections import OrderedDict
import datetime

class SEIR_Testing():

    def __init__(self, R0=2.2, T_inf=2.9, T_inc=5.2, T_hosp=5, T_death=32, P_severe=0.2, P_fatal=0.02, T_recov_severe=14,
                 T_recov_mild=11, N=7e6, init_infected=1, intervention_day=100, intervention_amount=0.33, q=0,
                 testing_rate_for_exposed=0, positive_test_rate_for_exposed=1, testing_rate_for_infected=0,
                 positive_test_rate_for_infected=1, intervention_removal_day=45, starting_date='2020-03-09'):

        T_trans = T_inf/R0
        T_recov_mild = (14 - T_inf)
        T_recov_severe = (31.5 - T_inf)

        P_mild = 1 - P_severe - P_fatal

        # define testing related parameters
        T_inf_detected = T_inf
        T_trans_detected = T_trans
        T_inc_detected = T_inc

        P_mild_detected = P_mild
        P_severe_detected = P_severe
        P_fatal_detected = P_fatal

        vanilla_params = {

            'R0': R0,

            'T_trans': T_trans,
            'T_inc': T_inc,
            'T_inf': T_inf,

            'T_recov_mild': T_recov_mild,
            'T_recov_severe': T_recov_severe,
            'T_hosp': T_hosp,
            'T_death': T_death,

            'P_mild': P_mild,
            'P_severe': P_severe,
            'P_fatal': P_fatal,
            'intervention_day': intervention_day,
            'intervention_removal_day': intervention_removal_day,
            'intervention_amount': intervention_amount,
            'starting_date': starting_date,
            'N': N
        }

        testing_params = {
            'T_trans': T_trans_detected,
            'T_inc': T_inc_detected,
            'T_inf': T_inf_detected,

            'P_mild': P_mild_detected,
            'P_severe': P_severe_detected,
            'P_fatal': P_fatal_detected,

            'q': q,
            'testing_rate_for_exposed': testing_rate_for_exposed,
            'positive_test_rate_for_exposed': positive_test_rate_for_exposed,
            'testing_rate_for_infected': testing_rate_for_infected,
            'positive_test_rate_for_infected': positive_test_rate_for_infected
        }

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

        # self.vanilla_params = vanilla_params
        # self.testing_params = testing_params
        # self.state_init_values = state_init_values
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
            self.R0 = self.intervention_amount * self.R0
            self.T_trans = self.T_inf/self.R0

        # Modelling the intervention
        if t >= self.intervention_removal_day:
            self.R0 = self.R0 / self.intervention_amount
            self.T_trans = self.T_inf/self.R0

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
        # import pdb; pdb.set_trace()
        df_prediction['date'] = pd.date_range(self.starting_date, self.starting_date + datetime.timedelta(days=df_prediction.shape[0] - 1))
        columns = list(df_prediction.columns)
        columns.remove('date')
        df_prediction = df_prediction[['date'] + columns]

        df_prediction['hospitalisations'] = df_prediction['R_severe_home'] + df_prediction['R_severe_hosp'] + df_prediction['R_fatal']
        df_prediction['recoveries'] = df_prediction['C']
        df_prediction['fatalities'] = df_prediction['D']
        df_prediction['infectious_unknown'] = df_prediction['I'] + df_prediction['D_I']
        df_prediction['total_infected'] = df_prediction['hospitalisations'] + df_prediction['recoveries'] + df_prediction['fatalities']
        return df_prediction