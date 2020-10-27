import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

from collections import OrderedDict
import datetime
import copy

from models.seir.seir import SEIR
from utils.fitting.ode import ODE_Solver

class SEIR_Undetected_Testing(SEIR):
    def __init__(self, lockdown_R0=2.2, T_inf_D=3.3, T_inf_U = 5.5, T_inc=5, T_recov_fatal=32,
                 P_fatal=0.2, T_recov_severe=14, N=1e7, d=1.0, psi=1.00, beta=0.1, starting_date='2020-03-09', 
                 observed_values=None, E_hosp_ratio=0.5, I_D_hosp_ratio=0.5, I_U_hosp_ratio=0.5, **kwargs):
        """
        This class implements SEIR + Hospitalisation + Severity Levels 

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
        d: Current Detection Ratio
        psi: effective sensititivity (based on antigen and rtpcr sensitive and their overall proportion)
        """
        STATES = ['S', 'E', 'I_D', 'I_U', 'P_U', 'R_severe', 'R_fatal', 'C', 'D']
        R_STATES = [x for x in STATES if 'R_' in x]
        input_args = copy.deepcopy(locals())
        # import pdb; pdb.set_trace()
        del input_args['self']
        p_params = {k: input_args[k] for k in input_args.keys() if 'P_' in k}
        t_params = {k: input_args[k] for k in input_args.keys() if 'T_recov' in k}
        p_params['P_severe'] = 1 - p_params['P_fatal']
        input_args['p_params'] = p_params
        input_args['t_params'] = t_params
        input_args['I_hosp_ratio'] = I_D_hosp_ratio + I_U_hosp_ratio
        self.daily_tests = input_args['kwargs']['daily_testing'] / N 
        del input_args['kwargs']
        super().__init__(**input_args)

        # import pdb; pdb.set_trace()
        self.d = d
        self.psi = psi
        self.T_inf_D = T_inf_D
        self.T_inf_U = T_inf_U
        self.I_D_hosp_ratio = I_D_hosp_ratio
        self.I_U_hosp_ratio = I_U_hosp_ratio
        self.beta = beta
        # self.state_init_values['I_D'] = observed_values['i_d'] / self.N
        # self.state_init_values['I_U'] = observed_values['i_u'] / self.N
        # self.state_init_values['P_U'] = observed_values['p_i'] / self.N
        # self.state_init_values['E'] = observed_values['e'] / self.N
        # self.state_init_values['S'] = observed_values['s'] / self.N
        # self.state_init_values['R_severe'] = observed_values['r_severe'] / self.N
        # self.state_init_values['R_fatal'] = observed_values['r_fatal'] / self.N
        # import pdb; pdb.set_trace()
        self.state_init_values['I_D'] = self.I_D_hosp_ratio * observed_values['active'] / self.N
        self.state_init_values['I_U'] = self.I_U_hosp_ratio * observed_values['active'] / self.N
        del self.state_init_values['I']


    def get_derivative(self, t, y):
        """
        Calculates derivative at time t
        """
        # Init state variables
        for i, _ in enumerate(y):
            y[i] = max(y[i], 0)
        S, E, I_D, I_U, P_U, R_severe, R_fatal, C, D = y

        # Init derivative vector
        
        dydt = np.zeros(y.shape)
        try:
            tests_done = self.daily_tests[self.starting_date + pd.Timedelta(days=max(1,math.ceil(t)))]
        except:
            tests_done = self.daily_tests[-1]

        # Write differential equations
        dydt[0] = - (I_D + I_U) * S * self.beta  # S
        dydt[1] = (I_D + I_U) * S * self.beta - (E/ self.T_inc)  # E
        dydt[2] = (1 / self.T_inc)*(1000*tests_done*self.d*self.psi)*E - I_D / self.T_inf_D  # I_D
        dydt[3] = (1 / self.T_inc)*(1 - 1000*tests_done*self.d*self.psi)*E - I_U / self.T_inf_U  # I_U
        dydt[4] = I_U / self.T_inf_U  # P_U
        dydt[5] = (1/self.T_inf_D)*(self.P_severe*I_D) - R_severe/self.T_recov_severe #R_severe
        dydt[6] = (1/self.T_inf_D)*(self.P_fatal*I_D) - R_fatal/self.T_recov_fatal # R_fatal
        dydt[7] = R_severe/self.T_recov_severe   # C
        dydt[8] = R_fatal/self.T_recov_fatal # D

        return dydt

    def predict(self, total_days=50, time_step=1, method='Radau'):
        """
        Returns predictions of the model
        """
        # Solve ODE get result
        df_prediction = super().predict(total_days=total_days,
                                        time_step=time_step, method=method)

        df_prediction['active'] =  df_prediction['R_severe'] + df_prediction['R_fatal']
        df_prediction['recovered'] = df_prediction['C']
        df_prediction['deceased'] = df_prediction['D']
        df_prediction['total'] = df_prediction['active'] + \
            df_prediction['recovered'] + df_prediction['deceased']
        return df_prediction
