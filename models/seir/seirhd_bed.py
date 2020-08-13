import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from collections import OrderedDict
import datetime
import copy

from models.seir.seir import SEIR
from utils.ode import ODE_Solver

class SEIRHD_Bed(SEIR):
    def __init__(self, pre_lockdown_R0=3, lockdown_R0=2.2, post_lockdown_R0=None, T_inf=2.9, T_inc=5.2,
                 P_nonoxy=0.4, P_oxy=0.2, P_icu=0.02, P_vent=0.02, P_fatal=0.02, 
                 T_recov_hq=14, T_recov_non_oxy=14, T_recov_oxy=14, T_recov_icu=14, T_recov_vent=14, T_recov_fatal=14,
                 N=7e6, lockdown_day=10, lockdown_removal_day=75, starting_date='2020-03-09', 
                 observed_values=None, E_hosp_ratio=0.5, I_hosp_ratio=0.5, **kwargs):
        
        """
        This class implements SEIR + Hospitalisation + Severity Levels 
        The model further implements 
        - pre, post, and during lockdown behaviour 


        The state variables are : 

        S : No of susceptible people
        E : No of exposed people
        I : No of infected people
        R_hq : No of people recovering from a hq version of the infection
        R_nonoxy : No of people recovering from a nonoxy version of the infection
        R_oxy : No of people recovering from a oxy version of the infection
        R_icu : No of people recovering from a icu version of the infection
        R_vent : No of people recovering from a vent version of the infection
        R_fatal : No of people recovering from a fatal version of the infection
        C : No of recovered people
        D : No of deceased people 

        The sum total is is always N (total population)

        """

        """
        The parameters are : 

        R0 values - 
        pre_lockdown_R0: R0 value pre-lockdown (float)
        lockdown_R0: R0 value during lockdown (float)
        post_lockdown_R0: R0 value post-lockdown (float)

        Transmission parameters - 
        T_inc: The incubation time of the infection (float)
        T_inf: The duration for which an individual is infectious (float)

        Probability of contracting different types of infections - 
        P_HQ: Probability of contracting a HQ infection (float - [0, 1])
        P_nonoxy: Probability of contracting a nonoxy infection (float - [0, 1])
        P_oxy: Probability of contracting a oxy infection (float - [0, 1])
        P_icu: Probability of contracting a icu infection (float - [0, 1])
        P_vent: Probability of contracting a vent infection (float - [0, 1])
        P_fatal: Probability of contracting a fatal infection (float - [0, 1])

        Clinical time parameters - 
        T_recov_hq: Time it takes for an individual with a hq infection to recover (float)
        T_recov_non_oxy: Time it takes for an individual with a nonoxy infection to recover (float)
        T_recov_oxy: Time it takes for an individual with a oxy infection to recover (float)
        T_recov_icu: Time it takes for an individual with a icu infection to recover (float)
        T_recov_vent: Time it takes for an individual with a vent infection to recover (float)
        T_recov_fatal: Time it takes for an individual with a fatal infection to die (float)

        Lockdown parameters - 
        starting_date: Datetime value that corresponds to Day 0 of modelling (datetime/str)
        lockdown_day: Number of days from the starting_date, after which lockdown is initiated (int)
        lockdown_removal_day: Number of days from the starting_date, after which lockdown is removed (int)

        Misc - 
        N: Total population
        """
        STATES = ['S', 'E', 'I', 'R_hq', 'R_nonoxy', 'R_oxy', 'R_icu', 'R_vent', 'R_fatal', 'C', 'D']
        R_STATES = [x for x in STATES if 'R_' in x]
        input_args = copy.deepcopy(locals())
        del input_args['self']
        del input_args['kwargs']
        p_params = {k: input_args[k] for k in input_args.keys() if 'P_' in k}
        t_params = {k: input_args[k] for k in input_args.keys() if 'T_recov' in k}
        P_mild = 1 - sum(p_params.values())
        p_params['P_hq'] = P_mild
        input_args['p_params'] = p_params
        input_args['t_params'] = t_params
        super().__init__(**input_args)

        # Initialisation
        state_init_values = OrderedDict()
        for key in STATES:
            state_init_values[key] = 0
        if initialisation == 'starting':
            init_infected = max(observed_values['init_infected'], 1)
            state_init_values['S'] = (self.N - init_infected)/self.N
            state_init_values['I'] = init_infected/self.N

        if initialisation == 'intermediate':
            
            state_init_values['R_hq'] = observed_values['hq']
            state_init_values['R_nonoxy'] = observed_values['non_o2_beds']
            state_init_values['R_oxy'] = observed_values['o2_beds']
            state_init_values['R_icu'] = observed_values['icu']
            state_init_values['R_vent'] = observed_values['ventilator']
            state_init_values['R_fatal'] = p_params['P_fatal'] * observed_values['active']
            
            state_init_values['C'] = observed_values['recovered']
            state_init_values['D'] = observed_values['deceased']

            state_init_values['E'] = self.E_hosp_ratio * observed_values['active']
            state_init_values['I'] = self.I_hosp_ratio * observed_values['active']
            
            nonSsum = sum(state_init_values.values())
            state_init_values['S'] = (self.N - nonSsum)
            for key in state_init_values.keys():
                state_init_values[key] = state_init_values[key]/self.N
        
        self.state_init_values = state_init_values

    def get_derivative(self, t, y):
        """
        Calculates derivative at time t
        """
        # Init state variables
        for i, _ in enumerate(y):
            y[i] = max(y[i], 0)
        S, E, I, R_hq, R_nonoxy, R_oxy, R_icu, R_vent, R_fatal, C, D = y

        # Modelling the behaviour post-lockdown
        if t >= self.lockdown_removal_day:
            self.R0 = self.post_lockdown_R0
        # Modelling the behaviour lockdown
        elif t >= self.lockdown_day:
            self.R0 = self.lockdown_R0
        # Modelling the behaviour pre-lockdown
        else:
            self.R0 = self.pre_lockdown_R0

        self.T_trans = self.T_inf/self.R0

        # Init derivative vector
        dydt = np.zeros(y.shape)

        # Write differential equations
        dydt[0] = - I * S / (self.T_trans)  # S
        dydt[1] = I * S / (self.T_trans) - (E/ self.T_inc)  # E
        dydt[2] = E / self.T_inc - I / self.T_inf  # I
        dydt[3] = (1/self.T_inf)*(self.P_hq*I) - R_hq/self.T_recov_hq # R_hq
        dydt[4] = (1/self.T_inf)*(self.P_nonoxy*I) - R_nonoxy/self.T_recov_non_oxy #R_nonoxy
        dydt[5] = (1/self.T_inf)*(self.P_oxy*I) - R_oxy/self.T_recov_oxy #R_oxy
        dydt[6] = (1/self.T_inf)*(self.P_icu*I) - R_icu/self.T_recov_icu # R_icu
        dydt[7] = (1/self.T_inf)*(self.P_vent*I) - R_vent/self.T_recov_vent #R_vent
        dydt[8] = (1/self.T_inf)*(self.P_fatal*I) - R_fatal/self.T_recov_fatal # R_fatal
        dydt[9] = R_hq/self.T_recov_hq + R_nonoxy/self.T_recov_non_oxy + \
            R_oxy/self.T_recov_oxy + R_icu/self.T_recov_icu + R_vent/self.T_recov_vent  # C
        dydt[10] = R_fatal/self.T_recov_fatal # D

        return dydt

    def predict(self, total_days=50, time_step=1, method='Radau'):
        """
        Returns predictions of the model
        """
        # Solve ODE get result
        df_prediction = super().predict(total_days=total_days,
                                        time_step=time_step, method=method)

        df_prediction['hq'] = df_prediction['R_hq']
        df_prediction['non_o2_beds'] = df_prediction['R_nonoxy']
        df_prediction['o2_beds'] = df_prediction['R_oxy']
        df_prediction['icu'] = df_prediction['R_icu']
        df_prediction['ventilator'] = df_prediction['R_vent']
        df_prediction['active'] = df_prediction['hq'] + df_prediction['non_o2_beds'] + \
            df_prediction['o2_beds'] + df_prediction['icu'] + df_prediction['ventilator']
        df_prediction['recovered'] = df_prediction['C']
        df_prediction['deceased'] = df_prediction['D']
        df_prediction['total'] = df_prediction['active'] + df_prediction['recovered'] + df_prediction['deceased']
        return df_prediction
