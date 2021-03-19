import datetime
from abc import abstractmethod
from collections import OrderedDict

import pandas as pd

from models.seir.compartmental_base import CompartmentalBase
from utils.fitting.ode import ODE_Solver


class SEIRBase(CompartmentalBase):

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
        return super().predict(total_days=total_days, time_step=time_step, method=method)
