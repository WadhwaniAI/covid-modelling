import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from collections import OrderedDict

class SEIR_Testing():

    def __init__(self, R0=3, T_inf=3.3, T_inc=4.5, T_hosp=5, 
                 T_death=32, P_severe=0.8, P_fatal=0.11, T_recov_severe=26, T_recov_mild=11, N=7e6, init_infected=1,
                 q=0, theta_E=0, psi_E=1, theta_I=0, psi_I=1, initialisation='starting', observed_values=None, 
                 E_hosp_ratio=0.5, I_hosp_ratio=0.39, factor=(1.77/1.23), int_vec=None, ** kwargs):
        """
        This class implements SEIR + Hospitalisation + Severity Levels + Testing 
        The model further implements 
        - pre, post, and during lockdown behaviour 
        - different initialisations : intermediate and starting 

        The state variables are : 

        S : No of susceptible people
        E : No of exposed people
        I : No of infected people
        D_E : No of exposed people (detected)
        D_I : No of infected people (detected)
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

        Testing Parameters - 
        'q': Perfection of quarantining 
        If q = 0, quarantining is perfect. q = 1. quarantining is absolutely imperfect
        'theta_E': Percentage of people in the Exposed bucket that are tested daily
        'psi_E': Sensitivity of test that Exposed people undergo
        'theta_I': Percentage of people in the Infected bucket that are tested daily
        'psi_I': Sensitivity of test that Infected people undergo

        Lockdown parameters - 
        starting_date: Datetime value that corresponds to Day 0 of modelling (datetime/str)
        lockdown_day: Number of days from the starting_date, after which lockdown is initiated (int)
        lockdown_removal_day: Number of days from the starting_date, after which lockdown is removed (int)

        Misc - 
        N: Total population
        initialisation : method of initialisation ('intermediate'/'starting')
        E_hosp_ratio : Ratio for Exposed to hospitalised for initialisation
        I_hosp_ratio : Ratio for Infected to hospitalised for initialisation
        """

        P_mild = 1 - P_severe - P_fatal

        # define testing related parameters
        T_inf_detected = T_inf
        T_inc_detected = T_inc

        P_mild_detected = P_mild
        P_severe_detected = P_severe
        P_fatal_detected = P_fatal

        vanilla_params = {
            # R0 values
            'R0': R0, # R0 value pre-lockdown

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

            'N': N,
            'factor': factor,
            'int_vec': int_vec,
            #Initialisation Params
            'E_hosp_ratio' : E_hosp_ratio, # Ratio for Exposed to hospitalised for initialisation
            'I_hosp_ratio' : I_hosp_ratio # Ratio for Infected to hospitalised for initialisation
        }

        testing_params = {
            'T_inc': T_inc_detected,
            'T_inf': T_inf_detected,

            'P_mild': P_mild_detected,
            'P_severe': P_severe_detected,
            'P_fatal': P_fatal_detected,

            # Testing Parameters
            'q': q, # Perfection of quarantining : If q = 0, quarantining is perfect. q = 1. quarantining is absolutely imperfect
            'theta_E': theta_E, # Percentage of people in the Exposed bucket that are tested daily
            'psi_E': psi_E, # Sensitivity of test that Exposed people undergo
            'theta_I': theta_I, # Percentage of people in the Infected bucket that are tested daily
            'psi_I': psi_I # Sensitivity of test that Infected people undergo
        }

         # Set all dicts as attributes of self
        for param_dict_name in ['vanilla_params', 'testing_params']:
            setattr(self, param_dict_name, eval(param_dict_name))

        # Set all variables as attributes of self
        for key in self.vanilla_params:
            setattr(self, key, self.vanilla_params[key])

        for key in self.testing_params:
            suffix = '_D' if key in self.vanilla_params else ''
            setattr(self, key + suffix, self.testing_params[key])

        # Initialisation
        state_init_values = OrderedDict()
        key_order = ['S', 'E', 'I', 'D_E', 'D_I', 'R_mild', 'R_severe_home', 'R_severe_hosp', 'R_fatal', 'C', 'D']
        for key in key_order:
            state_init_values[key] = 0
        
        if initialisation == 'starting':
            # init_infected = max(observed_values['init_infected'], 1)
            init_infected = 0.001 * self.N
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

    def get_impact(self, choice, factor):
            return(1+factor*choice)

    def get_derivative(self, t, y):
        """
        Calculates derivative at time t
        """

        # Init state variables
        for i, _ in enumerate(y):
            y[i] = max(y[i], 0)
        [S, E, I, D_E, D_I, R_mild, R_severe_home, R_severe_hosp, R_fatal, C, D] = y

        self.T_base = self.T_inf/self.R0
        self.T_base_D = self.T_inf_D/self.R0

        # Modelling the intervention
        try:
            self.T_trans = self.get_impact(self.int_vec[t],self.factor)*self.T_base
        except:
            self.T_trans = self.T_base

        try:
            self.T_trans_D = self.get_impact(self.int_vec[t],self.factor)*self.T_base_D
        except:
            self.T_trans_D = self.T_base_D

        # Init derivative vector
        dydt = np.zeros(len(y))

        # Write differential equations
        dydt[0] = - I * S / (self.T_trans) - (self.q / self.T_trans_D) * (S * D_I) # S
        dydt[1] = I * S / (self.T_trans) + (self.q / self.T_trans_D) * (S * D_I) - (E/ self.T_inc) - (self.theta_E * self.psi_E * E) # E
        dydt[2] = E / self.T_inc - I / self.T_inf - (self.theta_I * self.psi_I * I) # I
        dydt[3] = (self.theta_E * self.psi_E * E) - (1 / self.T_inc_D) * D_E # D_E
        dydt[4] = (self.theta_I * self.psi_I * I) + (1 / self.T_inc_D) * D_E - (1 / self.T_inf_D) * D_I # D_I 
        dydt[5] = (1/self.T_inf)*(self.P_mild*I) + (1/self.T_inf_D)*(self.P_mild_D*D_I) - R_mild/self.T_recov_mild # R_mild
        dydt[6] = (1/self.T_inf)*(self.P_severe*I) + (1/self.T_inf_D)*(self.P_severe_D*D_I) - R_severe_home/self.T_hosp # R_severe_home
        dydt[7] = R_severe_home/self.T_hosp - R_severe_hosp/self.T_recov_severe # R_severe_hosp
        dydt[8] = (1/self.T_inf)*(self.P_fatal*I) + (1/self.T_inf_D)*(self.P_fatal_D*D_I) - R_fatal/self.T_death # R_fatal
        dydt[9] = R_mild/self.T_recov_mild + R_severe_hosp/self.T_recov_severe # C
        dydt[10] = R_fatal/self.T_death # D

        return dydt

    def solve_ode(self, total_no_of_days=200, time_step=1, method='Radau'):
        """
        Solves ODE
        """
        # key_order = ['S', 'E', 'I', 'D_E', 'D_I', 'R_mild', 'R_severe_home', 'R_severe_hosp', 'R_fatal', 'C', 'D']
        t_start = 0
        t_final = total_no_of_days
        time_steps = np.arange(t_start, total_no_of_days, time_step)

        S_array = np.ones(len(time_steps))
        E_array = np.ones(len(time_steps))
        I_array = np.ones(len(time_steps))
        D_E_array = np.ones(len(time_steps))
        D_I_array = np.ones(len(time_steps))
        R_mild_array = np.ones(len(time_steps))
        R_severe_home_array = np.ones(len(time_steps))
        R_severe_hosp_array = np.ones(len(time_steps))
        R_fatal_array = np.ones(len(time_steps))
        C_array = np.ones(len(time_steps))
        D_array = np.ones(len(time_steps))

        state_init_values_arr = [self.state_init_values[x] for x in self.state_init_values]
        [S, E, I, D_E, D_I, R_mild, R_severe_home, R_severe_hosp, R_fatal, C, D] = state_init_values_arr

        for i in range(len(time_steps)):
            S_array[i], E_array[i], I_array[i], D_E_array[i], D_I_array[i], R_mild_array[i], R_severe_home_array[i], R_severe_hosp_array[i], R_fatal_array[i],\
                                                C_array[i], D_array[i] = S, E, I, D_E, D_I, R_mild, R_severe_home, R_severe_hosp, R_fatal, C, D
            t = time_steps[i]
            dydt = self.get_derivative(t,[S, E, I, D_E, D_I, R_mild, R_severe_home, R_severe_hosp, R_fatal, C, D])
            S += dydt[0]
            E += dydt[1]
            I += dydt[2]
            D_E += dydt[3]
            D_I += dydt[4]
            R_mild += dydt[5]
            R_severe_home += dydt[6]
            R_severe_hosp += dydt[7]
            R_fatal += dydt[8]
            C += dydt[9]
            D += dydt[10]
        sol = np.array([S_array, E_array, I_array, D_E_array, D_I_array, R_mild_array, R_severe_home_array, R_severe_hosp_array, R_fatal_array,C_array, D_array])
        
        return sol


    def return_predictions(self, sol):
        """
        Returns predictions of the model
        sol : Solved ODE variable
        """
        states_time_matrix = (sol.y*self.vanilla_params['N']).astype('int')
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
        df_prediction['infectious_unknown'] = df_prediction['I'] + df_prediction['D_I']
        df_prediction['total_infected'] = df_prediction['hospitalised'] + df_prediction['recovered'] + df_prediction['deceased']
        return df_prediction
