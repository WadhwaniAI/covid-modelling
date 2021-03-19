import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from collections import OrderedDict
import datetime
import copy

from models.seir.seir_base import SEIR
from utils.fitting.ode import ODE_Solver

class SEIRHD_t(SEIR):
    def __init__(self, lockdown_R0=2.2, T_inf=2.9, T_inc=5.2, T_recov_fatal=32,
                 P_severe=0.2, P_fatal=0.02, T_recov_severe=14, T_recov_mild=11, N=7e6,
                 starting_date='2020-03-09',
                 time_varying_start_date='2020-06-10', time_varying_length=10, factor_to_increase_gamma=2, 
                 observed_values=None, E_hosp_ratio=0.5, I_hosp_ratio=0.5, **kwargs):
        """
        This class implements SEIR + Hospitalisation + Severity Levels 
        The model further implements 
        - time varying parameters

        The state variables are : 

        S : No of susceptible people
        E : No of exposed people
        I : No of infected people
        R_mild : No of people recovering from a mild version of the infection
        R_severe : No of people recovering from a severe version of the infection
        R_fatal : No of people recovering from a fatal version of the infection
        C : No of recovered people
        D : No of deceased people 

        The sum total is is always N (total population)

        """

        """
        The parameters are : 

        R0 values - 
        lockdown_R0: R0 value during lockdown (float)

        Transmission parameters - 
        T_inc: The incubation time of the infection (float)
        T_inf: The duration for which an individual is infectious (float)

        Probability of contracting different types of infections - 
        P_mild: Probability of contracting a mild infection (float - [0, 1])
        P_severe: Probability of contracting a severe infection (float - [0, 1])
        P_fatal: Probability of contracting a fatal infection (float - [0, 1])

        Clinical time parameters - 
        T_recov_mild: Time it takes for an individual with a mild infection to recover (float)
        T_recov_severe: Time it takes for an individual with a severe infection to recover (float)
        T_recov_fatal: Time it takes for an individual with a fatal infection to die (float)

        Lockdown parameters - 
        starting_date: Datetime value that corresponds to Day 0 of modelling (datetime/str)

        Misc - 
        N: Total population
        """
        STATES = ['S', 'E', 'I', 'R_mild', 'R_severe', 'R_fatal', 'C', 'D']
        R_STATES = [x for x in STATES if 'R_' in x]
        input_args = copy.deepcopy(locals())
        del input_args['self']
        del input_args['kwargs']
        p_params = {k: input_args[k] for k in input_args.keys() if 'P_' in k}
        t_params = {k: input_args[k] for k in input_args.keys() if 'T_recov' in k}
        p_params['P_severe'] = 1 - p_params['P_fatal']
        p_params['P_mild'] = 0
        input_args['p_params'] = p_params
        input_args['t_params'] = t_params
        super().__init__(**input_args)
        time_varying_start_date = datetime.datetime.strptime(time_varying_start_date, '%Y-%m-%d')
        extra_args = {
            'beta' : self.lockdown_R0/self.T_inf,
            'time_varying_start_date': time_varying_start_date,
            'days_to_time_varying': (time_varying_start_date - starting_date).days,
            'T_inf_arr' : np.linspace(T_inf, T_inf/factor_to_increase_gamma, time_varying_length),
            'factor_to_increase_gamma': factor_to_increase_gamma,
            'time_varying_length': time_varying_length
        }
        for key in extra_args:
            setattr(self, key, extra_args[key])


    def get_derivative(self, t, y):
        """
        Calculates derivative at time t
        """
        # Init state variables
        for i, _ in enumerate(y):
            y[i] = max(y[i], 0)
        S, E, I, R_mild, R_severe, R_fatal, C, D = y

        if (t >= self.days_to_time_varying) & (t < (self.days_to_time_varying + self.time_varying_length)):
            T_inf = self.T_inf_arr[int(t - self.days_to_time_varying)]
            self.R0 = T_inf*self.beta

        if t >= (self.days_to_time_varying + self.time_varying_length):
            self.T_inf = self.T_inf_arr[-1]
            self.R0 = self.T_inf*self.beta
        
        if t < self.days_to_time_varying:
            self.R0 = self.lockdown_R0
        self.T_trans = self.T_inf/self.R0


        # Init derivative vector
        dydt = np.zeros(y.shape)

        # Write differential equations
        dydt[0] = - I * S / self.T_trans  # S
        dydt[1] = I * S / self.T_trans - (E / self.T_inc)  # E
        dydt[2] = E / self.T_inc - I / self.T_inf  # I
        dydt[3] = (1/self.T_inf)*(self.P_mild*I) - R_mild/self.T_recov_mild # R_mild
        dydt[4] = (1/self.T_inf)*(self.P_severe*I) - R_severe/self.T_recov_severe #R_severe
        dydt[5] = (1/self.T_inf)*(self.P_fatal*I) - R_fatal/self.T_recov_fatal # R_fatal
        dydt[6] = R_mild/self.T_recov_mild + R_severe/self.T_recov_severe  # C
        dydt[7] = R_fatal/self.T_recov_fatal # D

        return dydt

    def predict(self, total_days=50, time_step=1, method='Radau'):
        """
        Returns predictions of the model
        """
        # Solve ODE get result
        df_prediction = super().predict(total_days=total_days,
                                        time_step=time_step, method=method)

        df_prediction['active'] = df_prediction['R_mild'] + \
            df_prediction['R_severe'] + df_prediction['R_fatal']
        df_prediction['recovered'] = df_prediction['C']
        df_prediction['deceased'] = df_prediction['D']
        df_prediction['total'] = df_prediction['active'] + \
            df_prediction['recovered'] + df_prediction['deceased']
        return df_prediction
