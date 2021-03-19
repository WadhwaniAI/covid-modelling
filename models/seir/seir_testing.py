import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from collections import OrderedDict
import datetime
import copy

from models.seir.seir_base import SEIRBase
from utils.fitting.ode import ODE_Solver

class SEIR_Testing(SEIRBase):

    def __init__(self, lockdown_R0=2.2, T_inf=2.9, T_inc=5.2,
                 T_recov_fatal=32, P_severe=0.2, P_fatal=0.02, T_recov_severe=14, T_recov_mild=11, N=7e6,
                 q=0, theta_E=0, psi_E=1, theta_I=0, psi_I=1,
                 starting_date='2020-03-09', observed_values=None, 
                 E_hosp_ratio=0.5, I_hosp_ratio=0.5, ** kwargs):
        """
        This class implements SEIR + Hospitalisation + Severity Levels + Testing 

        The state variables are : 

        S : No of susceptible people
        E : No of exposed people
        I : No of infected people
        D_E : No of exposed people (detected)
        D_I : No of infected people (detected)
        R_mild : No of people recovering from a mild version of the infection
        R_severe : No of people recovering from a fatal version of the infection (at hospital)
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

        Testing Parameters - 
        'q': Perfection of quarantining 
        If q = 0, quarantining is perfect. q = 1. quarantining is absolutely imperfect
        'theta_E': Percentage of people in the Exposed bucket that are tested daily
        'psi_E': Sensitivity of test that Exposed people undergo
        'theta_I': Percentage of people in the Infected bucket that are tested daily
        'psi_I': Sensitivity of test that Infected people undergo

        Lockdown parameters - 
        starting_date: Datetime value that corresponds to Day 0 of modelling (datetime/str)

        Misc - 
        N: Total population
        E_hosp_ratio : Ratio for Exposed to hospitalised for initialisation
        I_hosp_ratio : Ratio for Infected to hospitalised for initialisation
        """
        STATES = ['S', 'E', 'I', 'D_E', 'D_I', 'R_mild', 'R_severe', 'R_fatal', 'C', 'D']
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

        extra_params = {
            # Testing Parameters
            'q': q, # Perfection of quarantining : If q = 0, quarantining is perfect. q = 1. quarantining is absolutely imperfect
            'theta_E': theta_E, # Percentage of people in the Exposed bucket that are tested daily
            'psi_E': psi_E, # Sensitivity of test that Exposed people undergo
            'theta_I': theta_I, # Percentage of people in the Infected bucket that are tested daily
            'psi_I': psi_I # Sensitivity of test that Infected people undergo
        }

        # Set all variables as attributes of self
        for key in extra_params:
            setattr(self, key, extra_params[key])

    def get_derivative(self, t, y):
        """
        Calculates derivative at time t
        """

        # Init state variables
        for i, _ in enumerate(y):
            y[i] = max(y[i], 0)
        S, E, I, D_E, D_I, R_mild, R_severe, R_fatal, C, D = y

        self.T_trans = self.T_inf/self.lockdown_R0

        # Init derivative vector
        dydt = np.zeros(y.shape)

        # Write differential equations
        dydt[0] = - ((I + self.q * D_I) * S) / self.T_trans  # S
        dydt[1] = ((I + self.q * D_I) * S ) / self.T_trans - (E / self.T_inc) - (self.theta_E * self.psi_E * E)  # E
        dydt[2] = E / self.T_inc - I / self.T_inf - (self.theta_I * self.psi_I * I) # I
        dydt[3] = (self.theta_E * self.psi_E * E) - (1 / self.T_inc) * D_E # D_E
        dydt[4] = (self.theta_I * self.psi_I * I) + (1 / self.T_inc) * D_E - (1 / self.T_inf) * D_I # D_I 
        dydt[5] = (1/self.T_inf)*(self.P_mild*(I + D_I)) - R_mild/self.T_recov_mild # R_mild
        dydt[6] = (1/self.T_inf)*(self.P_severe*(I + D_I)) - R_severe/self.T_recov_severe # R_severe
        dydt[7] = (1/self.T_inf)*(self.P_fatal*(I + D_I)) - R_fatal/self.T_recov_fatal # R_fatal
        dydt[8] = R_mild/self.T_recov_mild + R_severe/self.T_recov_severe # C
        dydt[9] = R_fatal/self.T_recov_fatal # D

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
