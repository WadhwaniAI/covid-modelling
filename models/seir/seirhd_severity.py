import numpy as np
from collections import OrderedDict
import copy

from models.seir.base import SEIR

class SEIRHD_Severity(SEIR):
    def __init__(self, lockdown_R0=2.2, T_inf=2.9, T_inc=5.2, #Transmission
                 P_moderate=0.4, P_severe=0.2, P_fatal=0.02,  #Clinical Probabs
                 T_recov_severe=14, T_recov_mild=11, T_recov_moderate=11, T_recov_fatal=32, #Clinical Time
                 N=7e6, starting_date='2020-03-09', #Misc
                 observed_values=None, E_hosp_ratio=0.5, I_hosp_ratio=0.5, #Init
                 super_init=False, **kwargs): #Init
        """
        This class implements SEIR + Hospitalisation + Severity Levels 

        The state variables are : 

        S : No of susceptible people
        E : No of exposed people
        I : No of infected people
        R_mild : No of people recovering from a mild version of the infection
        R_moderate : No of people recovering from a moderate version of the infection
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
        P_moderate: Probability of contracting a moderate infection (float - [0, 1])
        P_severe: Probability of contracting a severe infection (float - [0, 1])
        P_fatal: Probability of contracting a fatal infection (float - [0, 1])

        Clinical time parameters - 
        T_recov_mild: Time it takes for an individual with a mild infection to recover (float)
        T_recov_moderate: Time it takes for an individual with a moderate infection to recover (float)
        T_recov_severe: Time it takes for an individual with a severe infection to recover (float)
        T_recov_fatal: Time it takes for an individual with a fatal infection to die (float)

        Lockdown parameters - 
        starting_date: Datetime value that corresponds to Day 0 of modelling (datetime/str)

        Misc - 
        N: Total population
        """
        STATES = ['S', 'E', 'I', 'R_mild', 'R_moderate', 'R_severe', 'R_fatal', 'C', 'D']
        R_STATES = [x for x in STATES if 'R_' in x]
        input_args = copy.deepcopy(locals())
        del input_args['self']
        del input_args['kwargs']
        p_params = {k: input_args[k] for k in input_args.keys() if 'P_' in k}
        t_params = {k: input_args[k] for k in input_args.keys() if 'T_recov' in k}
        P_mild = 1 - sum(p_params.values())
        p_params['P_mild'] = P_mild
        input_args['p_params'] = p_params
        input_args['t_params'] = t_params
        super().__init__(**input_args)

        # Initialisation
        state_init_values = OrderedDict()
        for key in STATES:
            state_init_values[key] = 0
        
        state_init_values['R_mild'] = observed_values['asymptomatic']
        state_init_values['R_moderate'] = observed_values['symptomatic']
        state_init_values['R_severe'] = observed_values['critical']
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
        S, E, I, R_mild, R_moderate, R_severe, R_fatal, C, D = y

        self.T_trans = self.T_inf/self.lockdown_R0

        # Init derivative vector
        dydt = np.zeros(y.shape)

        # Write differential equations
        dydt[0] = - I * S / self.T_trans  # S
        dydt[1] = I * S / self.T_trans - (E/ self.T_inc)  # E
        dydt[2] = E / self.T_inc - I / self.T_inf  # I
        dydt[3] = (1/self.T_inf)*(self.P_mild*I) - R_mild/self.T_recov_mild # R_mild
        dydt[4] = (1/self.T_inf)*(self.P_moderate*I) - R_moderate/self.T_recov_moderate #R_moderate
        dydt[5] = (1/self.T_inf)*(self.P_severe*I) - R_severe/self.T_recov_severe #R_severe
        dydt[6] = (1/self.T_inf)*(self.P_fatal*I) - R_fatal/self.T_recov_fatal # R_fatal
        dydt[7] = R_mild/self.T_recov_mild + R_moderate/self.T_recov_moderate + R_severe/self.T_recov_severe  # C
        dydt[8] = R_fatal/self.T_recov_fatal # D

        return dydt

    def predict(self, total_days=50, time_step=1, method='Radau'):
        """
        Returns predictions of the model
        """
        # Solve ODE get result
        df_prediction = super().predict(total_days=total_days,
                                        time_step=time_step, method=method)

        df_prediction['active'] = df_prediction['R_mild'] + \
            df_prediction['R_moderate'] + df_prediction['R_severe']
        df_prediction['asymptomatic'] = df_prediction['R_mild']
        df_prediction['symptomatic'] = df_prediction['R_moderate']
        df_prediction['critical'] = df_prediction['R_severe']
        df_prediction['recovered'] = df_prediction['C']
        df_prediction['deceased'] = df_prediction['D']
        df_prediction['total'] = df_prediction['active'] + df_prediction['recovered'] + df_prediction['deceased']
        return df_prediction
