from abc import abstractmethod
from models.model import Model

import pandas as pd

from collections import OrderedDict
import datetime

from utils.fitting.ode import ODE_Solver

class SEIR(Model):

    @abstractmethod
    def __init__(self, STATES, R_STATES, p_params, t_params, lockdown_R0=2.2, T_inf=2.9, T_inc=5.2, N=7e6, 
                 starting_date='2020-03-09', observed_values=None, E_hosp_ratio=0.5, I_hosp_ratio=0.5, **kwargs):

        params = {
            # R0 values
            'lockdown_R0': lockdown_R0,  # R0 value during lockdown

            # Transmission parameters
            'T_inc': T_inc,  # The incubation time of the infection
            'T_inf': T_inf,  # The duration for which an individual is infectious

            # Lockdown parameters
            'starting_date': starting_date,  # Datetime value that corresponds to Day 0 of modelling
            'N': N,

            #Initialisation Params
            'E_hosp_ratio': E_hosp_ratio,  # Ratio for Exposed to hospitalised for initialisation
            'I_hosp_ratio': I_hosp_ratio  # Ratio for Infected to hospitalised for initialisation
        }

        for key in params:
            setattr(self, key, params[key])

        for key in p_params:
            setattr(self, key, p_params[key])

        for key in t_params:
            setattr(self, key, t_params[key])

        # Initialisation
        state_init_values = OrderedDict()
        for key in STATES:
            state_init_values[key] = 0

        for state in R_STATES:
            statename = state.split('R_')[1]
            P_keyname = [k for k in p_params.keys() if k.split('P_')[1] == statename][0]
            state_init_values[state] = p_params[P_keyname] * observed_values['active']
        
        state_init_values['C'] = observed_values['recovered']
        state_init_values['D'] = observed_values['deceased']

        state_init_values['E'] = self.E_hosp_ratio * observed_values['active']
        state_init_values['I'] = self.I_hosp_ratio * observed_values['active']
        
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