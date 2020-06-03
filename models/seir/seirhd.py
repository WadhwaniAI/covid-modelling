import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from collections import OrderedDict
import datetime

from models.seir.seir import SEIR

class SEIRHD(SEIR):
    def __init__(self, pre_lockdown_R0=3, lockdown_R0=2.2, post_lockdown_R0=None, T_inf=2.9, T_inc=5.2, T_hosp=5, 
                T_death=32, P_severe=0.2, P_fatal=0.02, T_recov_severe=14, T_recov_mild=11, N=7e6, init_infected=1,
                 lockdown_day=10, lockdown_removal_day=75, starting_date='2020-03-09', initialisation='intermediate', 
                 observed_values=None, E_hosp_ratio=0.5, I_hosp_ratio=0.5, **kwargs):
        """
        This class implements SEIR + Hospitalisation + Severity Levels 
        The model further implements 
        - pre, post, and during lockdown behaviour 
        - different initialisations : intermediate and starting 

        The state variables are : 

        S : No of susceptible people
        E : No of exposed people
        I : No of infected people
        R_mild : No of people recovering from a mild version of the infection
        R_severe_home : No of people recovering from a severe version of the infection (at home)
        R_severe_hosp : No of people recovering from a fatal version of the infection (at hospital)
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
        P_mild: Probability of contracting a mild infection (float - [0, 1])
        P_severe: Probability of contracting a severe infection (float - [0, 1])
        P_fatal: Probability of contracting a fatal infection (float - [0, 1])

        Clinical time parameters - 
        T_recov_mild: Time it takes for an individual with a mild infection to recover (float)
        T_recov_severe: Time it takes for an individual with a severe infection to recover (float)
        T_hosp: Time it takes for an individual to get hospitalised, after they have been diagnosed (float)
        T_death: Time it takes for an individual with a fatal infection to die (float)

        Lockdown parameters - 
        starting_date: Datetime value that corresponds to Day 0 of modelling (datetime/str)
        lockdown_day: Number of days from the starting_date, after which lockdown is initiated (int)
        lockdown_removal_day: Number of days from the starting_date, after which lockdown is removed (int)

        Misc - 
        N: Total population
        initialisation : method of initialisation ('intermediate'/'starting')
        """

        # If no value of post_lockdown R0 is provided, the model assumes the lockdown R0 post-lockdown
        if post_lockdown_R0 == None:
           post_lockdown_R0 = lockdown_R0

        # P_mild = 1 - P_severe - P_fatal
        P_severe = 1 - P_fatal
        P_mild = 0

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
            'lockdown_day': lockdown_day, # Number of days from the starting_date, after which lockdown is initiated
            'lockdown_removal_day': lockdown_removal_day, # Number of days from the starting_date, after which lockdown is removed
            'N': N
        }

        # Set all variables as attributes of self
        for key in self.vanilla_params:
            setattr(self, key, self.vanilla_params[key])

        # Initialisation
        state_init_values = OrderedDict()
        key_order = ['S', 'E', 'I', 'R_mild', 'R_severe_home', 'R_severe_hosp', 'R_fatal', 'C', 'D']
        for key in key_order:
            state_init_values[key] = 0
        if initialisation == 'starting':
            init_infected = max(observed_values['init_infected'], 1)
            state_init_values['S'] = (self.N - init_infected)/self.N
            state_init_values['I'] = init_infected/self.N

        if initialisation == 'intermediate':
            state_init_values['R_severe_hosp'] = self.P_severe / (self.P_severe + self.P_fatal) * observed_values['hospitalised']
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


    def get_derivative(self, t, y):
        """
        Calculates derivative at time t
        """
        # Init state variables
        for i, _ in enumerate(y):
            y[i] = max(y[i], 0)
        S, E, I, R_mild, R_severe_home, R_severe_hosp, R_fatal, C, D = y

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
        dydt[3] = (1/self.T_inf)*(self.P_mild*I) - R_mild/self.T_recov_mild # R_mild
        dydt[4] = (1/self.T_inf)*(self.P_severe*I) - R_severe_home/self.T_hosp # R_severe_home
        dydt[5] = R_severe_home/self.T_hosp - R_severe_hosp/self.T_recov_severe # R_severe_hosp
        dydt[6] = (1/self.T_inf)*(self.P_fatal*I) - R_fatal/self.T_death # R_fatal
        dydt[7] = R_mild/self.T_recov_mild + R_severe_hosp/self.T_recov_severe # C
        dydt[8] = R_fatal/self.T_death # D

        return dydt

    def solve_ode(self, total_no_of_days=200, time_step=1, method='Radau'):
        """
        Solves ODE
        """
        t_start = 0
        t_final = total_no_of_days
        time_steps = np.arange(t_start, total_no_of_days + time_step, time_step)
        
        state_init_values_arr = [self.state_init_values[x] for x in self.state_init_values]

        sol = solve_ivp(self.get_derivative, [t_start, t_final], 
                        state_init_values_arr, method=method, t_eval=time_steps)

        self.sol = sol

    def predict(self):
        """
        Returns predictions of the model
        """
        states_time_matrix = (self.sol.y*self.vanilla_params['N']).astype('int')
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
        df_prediction['infectious_unknown'] = df_prediction['I']
        df_prediction['total_infected'] = df_prediction['hospitalised'] + df_prediction['recovered'] + df_prediction['deceased']
        return df_prediction
