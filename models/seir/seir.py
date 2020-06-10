from abc import ABC, abstractmethod
from models import Model

import pandas as pd
import numpy as np

from collections import OrderedDict
import datetime

from utils.ode import ODE_Solver

class SEIR(Model):

    @abstractmethod
    def __init__(self, STATES, pre_lockdown_R0=3, lockdown_R0=2.2, post_lockdown_R0=None, T_inf=2.9, T_inc=5.2, 
                 T_death=32, P_severe=0.2, P_fatal=0.02, T_recov_severe=14, T_recov_mild=11, N=7e6,
                 lockdown_day=10, lockdown_removal_day=75, starting_date='2020-03-09', initialisation='intermediate',
                 observed_values=None, E_hosp_ratio=0.5, I_hosp_ratio=0.5, **kwargs):

        # If no value of post_lockdown R0 is provided, the model assumes the lockdown R0 post-lockdown
        if post_lockdown_R0 == None:
           post_lockdown_R0 = lockdown_R0

        # P_mild = 1 - P_severe - P_fatal
        P_severe = 1 - P_fatal
        P_mild = 0

        params = {
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
            'T_death': T_death, #Time it takes for an individual with a fatal infection to die

            # Lockdown parameters
            'starting_date': starting_date,  # Datetime value that corresponds to Day 0 of modelling
            'lockdown_day': lockdown_day, # Number of days from the starting_date, after which lockdown is initiated
            'lockdown_removal_day': lockdown_removal_day, # Number of days from the starting_date, after which lockdown is removed
            'N': N,

            #Initialisation Params
            'E_hosp_ratio': E_hosp_ratio,  # Ratio for Exposed to hospitalised for initialisation
            'I_hosp_ratio': I_hosp_ratio  # Ratio for Infected to hospitalised for initialisation
        }

        for key in params:
            setattr(self, key, params[key])

        # Initialisation
        state_init_values = OrderedDict()
        for key in STATES:
            state_init_values[key] = 0
        if initialisation == 'starting':
            init_infected = max(observed_values['init_infected'], 1)
            state_init_values['S'] = (self.N - init_infected)/self.N
            state_init_values['I'] = init_infected/self.N

        if initialisation == 'intermediate':
            state_init_values['R_severe'] = self.P_severe / (self.P_severe + self.P_fatal) * observed_values['hospitalised']
            state_init_values['R_fatal'] = self.P_fatal / (self.P_severe + self.P_fatal) * observed_values['hospitalised']
            state_init_values['C'] = observed_values['recovered']
            state_init_values['D'] = observed_values['deceased']

            state_init_values['E'] = self.E_hosp_ratio * observed_values['hospitalised']
            state_init_values['I'] = self.I_hosp_ratio * observed_values['hospitalised']
            
            nonSsum = sum(state_init_values.values())
            state_init_values['S'] = (self.N - nonSsum)
            for key in state_init_values.keys():
                state_init_values[key] = state_init_values[key]/self.N
        
        self.state_init_values = state_init_values
        
    
    @abstractmethod
    def get_derivative(self, t, y):
        pass

    @abstractmethod
    def predict(self, total_days, time_step, method):
        # Solve ODE get result
        solver = ODE_Solver()
        state_init_values_arr = [self.state_init_values[x]
                                 for x in self.state_init_values]

        sol = solver.solve_ode(state_init_values_arr, self.get_derivative, method=method, 
                               total_days=total_days, time_step=time_step)

        states_time_matrix = (sol.y*self.N).astype('int')

        dataframe_dict = {}
        for i, key in enumerate(self.state_init_values.keys()):
            dataframe_dict[key] = states_time_matrix[i]
        
        df_prediction = pd.DataFrame.from_dict(dataframe_dict)
        df_prediction['date'] = pd.date_range(self.starting_date, self.starting_date + datetime.timedelta(days=df_prediction.shape[0] - 1))
        columns = list(df_prediction.columns)
        columns.remove('date')
        df_prediction = df_prediction[['date'] + columns]
        return df_prediction
