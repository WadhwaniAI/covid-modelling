import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from collections import OrderedDict
import datetime
import pymc3 as pm
from pymc3.ode import DifferentialEquation
from theano.ifelse import ifelse
from theano import tensor as T, function, printing
import theano
theano.config.compute_test_value='ignore'
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

class SEIR_Testing():

    def __init__(self, R0=2.2, T_inf=2.9, T_inc=5.2, T_hosp=5, T_death=32, P_severe=0.2, P_fatal=0.02, T_recov_severe=14,
                 T_recov_mild=11, N=7e6, init_infected=1, intervention_day=100, intervention_amount=0.33, q=0,
                 testing_rate_for_exposed=0, positive_test_rate_for_exposed=1, testing_rate_for_infected=0,
                 positive_test_rate_for_infected=1, intervention_removal_day=45, starting_date='2020-03-09', 
                 state_init_values=None, **kwargs):

        T_trans = T_inf/R0
        T_recov_mild = (14 - T_inf)
        T_recov_severe = (31.5 - T_inf)

        P_mild = 1 - P_severe - P_fatal

        # define testing related parameters
        T_inf_detected = T_inf
        T_trans_detected = T_trans
        T_inc_detected = T_inc

        P_mild_detected = P_mild
        P_severe_detected = P_severe
        P_fatal_detected = P_fatal

        vanilla_params = {

            'R0': R0,

            'T_trans': T_trans,
            'T_inc': T_inc,
            'T_inf': T_inf,

            'T_recov_mild': T_recov_mild,
            'T_recov_severe': T_recov_severe,
            'T_hosp': T_hosp,
            'T_death': T_death,

            'P_mild': P_mild,
            'P_severe': P_severe,
            'P_fatal': P_fatal,
            'intervention_day': intervention_day,
            'intervention_removal_day': intervention_removal_day,
            'intervention_amount': intervention_amount,
            'starting_date': starting_date,
            'N': N
        }

        testing_params = {
            'T_trans': T_trans_detected,
            'T_inc': T_inc_detected,
            'T_inf': T_inf_detected,

            'P_mild': P_mild_detected,
            'P_severe': P_severe_detected,
            'P_fatal': P_fatal_detected,

            'q': q,
            'testing_rate_for_exposed': testing_rate_for_exposed,
            'positive_test_rate_for_exposed': positive_test_rate_for_exposed,
            'testing_rate_for_infected': testing_rate_for_infected,
            'positive_test_rate_for_infected': positive_test_rate_for_infected
        }

        if state_init_values == None:
            # S, E, D_E, D_I, I, R_mild, R_severe_home, R_severe_hosp, R_fatal, C, D
            state_init_values = OrderedDict()
            state_init_values['S'] = (N - init_infected)/N
            state_init_values['E'] = 0
            state_init_values['I'] = init_infected/N
            state_init_values['D_E'] = 0
            state_init_values['D_I'] = 0
            state_init_values['R_mild'] = 0
            state_init_values['R_severe_home'] = 0
            state_init_values['R_severe_hosp'] = 0
            state_init_values['R_fatal'] = 0
            state_init_values['C'] = 0
            state_init_values['D'] = 0

        # self.vanilla_params = vanilla_params
        # self.testing_params = testing_params
        # self.state_init_values = state_init_values
        for param_dict_name in ['vanilla_params', 'testing_params', 'state_init_values']:
            setattr(self, param_dict_name, eval(param_dict_name))


    def get_derivative(self, t, y):

        # Init state variables
        for i, _ in enumerate(y):
            y[i] = max(y[i], 0)
        S, E, I, D_E, D_I, R_mild, R_severe_home, R_severe_hosp, R_fatal, C, D = y

        # Init time parameters and probabilities
        for key in self.vanilla_params:
            setattr(self, key, self.vanilla_params[key])

        for key in self.testing_params:
            suffix = '_D' if key in self.vanilla_params else ''
            setattr(self, key + suffix, self.testing_params[key])

        # Modelling the intervention
        if t >= self.intervention_day:
            self.R0 = self.intervention_amount * self.R0
            self.T_trans = self.T_inf/self.R0

        # Modelling the intervention
        # if t >= self.intervention_removal_day:
        #     self.R0 = 1.2 * self.R0
        #     self.T_trans = self.T_inf/self.R0

        # Init derivative vector
        dydt = np.zeros(y.shape)
        
        theta_E = self.testing_rate_for_exposed
        psi_E = self.positive_test_rate_for_exposed
        theta_I = self.testing_rate_for_infected
        psi_I = self.positive_test_rate_for_infected

        # Write differential equations
        dydt[0] = - I * S / (self.T_trans) - (self.q / self.T_trans_D) * (S * D_I) # S
        dydt[1] = I * S / (self.T_trans) + (self.q / self.T_trans_D) * (S * D_I) - (E/ self.T_inc) - (theta_E * psi_E * E) # E
        dydt[2] = E / self.T_inc - I / self.T_inf - (theta_I * psi_I * I) # I
        dydt[3] = (theta_E * psi_E * E) - (1 / self.T_inc_D) * D_E # D_E
        dydt[4] = (theta_I * psi_I * I) + (1 / self.T_inc_D) * D_E - (1 / self.T_inf_D) * D_I # D_I 
        dydt[5] = (1/self.T_inf)*(self.P_mild*I) + (1/self.T_inf_D)*(self.P_mild_D*D_I) - R_mild/self.T_recov_mild # R_mild
        dydt[6] = (1/self.T_inf)*(self.P_severe*I) + (1/self.T_inf_D)*(self.P_severe_D*D_I) - R_severe_home/self.T_hosp # R_severe_home
        dydt[7] = R_severe_home/self.T_hosp - R_severe_hosp/self.T_recov_severe # R_severe_hosp
        dydt[8] = (1/self.T_inf)*(self.P_fatal*I) + (1/self.T_inf_D)*(self.P_fatal_D*D_I) - R_fatal/self.T_death # R_fatal
        dydt[9] = R_mild/self.T_recov_mild + R_severe_hosp/self.T_recov_severe # C
        dydt[10] = R_fatal/self.T_death # D

        return dydt

    def solve_ode(self, total_no_of_days=200, time_step=1, method='Radau'):
        t_start = 0
        t_final = total_no_of_days
        time_steps = np.arange(t_start, total_no_of_days + time_step, time_step)
        
        state_init_values_arr = [self.state_init_values[x] for x in self.state_init_values]

        sol = solve_ivp(self.get_derivative, [t_start, t_final], 
                        state_init_values_arr, method=method, t_eval=time_steps)

        return sol

    def return_predictions(self, sol):
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

class SEIR_Test_pymc3(SEIR_Testing):
    def __init__(self,  *args, **kwargs):
        super().__init__( *args, **kwargs)
    def get_derivative(self, y, t, p):
        # Init state variables
        #for i, _ in enumerate(y):
        #for i in range(11):
        #    y[i] = ifelse(T.lt(y[i], 0), y[i], np.float64(0))
        #    y[i] = max(y[i], 0)
        zero = T.cast(0.0, 'float64')
        for i in range(11):
            T.set_subtensor(y[i], ifelse(T.gt(y[i], zero), y[i], zero))
        # Init time parameters and probabilities
        for key in self.vanilla_params:
            setattr(self, key, self.vanilla_params[key])
        for key in self.testing_params:
            suffix = '_D' if key in self.vanilla_params else ''
            setattr(self, key + suffix, self.testing_params[key])
            
        
        ## Set up variables using `y` and `p`
        
        S = y[0]
        E = y[1]
        I = y[2]
        D_E = y[3]
        D_I = y[4]
        R_mild = y[5]
        R_severe_home = y[6]
        R_severe_hosp = y[7]
        R_fatal = y[8]
        C = y[9]
        D = y[10]
        
        # p
    
        self.R0 = p[0]
        self.T_inc = p[1]
        self.T_inf = p[2]
        self.T_recov_severe = p[3]
        self.P_severe = p[4]
        self.P_fatal = p[5]
        self.intervention_amount = p[6]
        
        #Define variables  
        #if self.post_lockdown_R0 == None:
        #    self.post_lockdown_R0 = self.lockdown_R0

        self.P_mild = 1 - self.P_severe - self.P_fatal

        # define testing related parameters
        self.T_inf_detected = self.T_inf
        self.T_inc_detected = self.T_inc

        self.P_mild_detected = self.P_mild
        self.P_severe_detected = self.P_severe
        self.P_fatal_detected = self.P_fatal
        #self.T_trans_D = self.T_trans
  
        self.theta_E = self.testing_rate_for_exposed
        self.psi_E = self.positive_test_rate_for_exposed
        self.theta_I = self.testing_rate_for_infected
        self.psi_I = self.positive_test_rate_for_infected
        #TODO incorporate lockdown R0 code
        #T.set_subtensor(self.R0, ifelse(T.gt(t, self.lockdown_removal_day), self.R0 , self.post_lockdown_R0))
        # Modelling the behaviour lockdown
        #elif t >= self.lockdown_day:
        #    self.R0 = self.lockdown_R0
        #T.set_subtensor(self.R0, ifelse(T.gt(t, self.lockdown_day), self.R0, self.lockdown_R0))
        # Modelling the behaviour pre-lockdown
        #else:
        #    self.R0 = self.pre_lockdown_R0
        #T.set_subtensor(self.R0, ifelse(T.gt(y[i], zero), self.R0, self.pre_lockdown_R0))
        self.T_trans = self.T_inf/self.R0
        self.T_trans_D = self.T_inf_D/self.R0
        
       
        # Write differential equations
        dS = - I * S / (self.T_trans) - (self.q / self.T_trans_D) * (S * D_I) # # S
        #dS = - y[2] * y[0]*p[0]/p[2]  - self.q*p[2] * (y[0] * y[4])
        dE = I * S / (self.T_trans) + (self.q / self.T_trans_D) * (S * D_I) - (E/ self.T_inc) - (self.theta_E * self.psi_E * E) # E
        dI = E / self.T_inc - I / self.T_inf - (self.theta_I * self.psi_I * I) # I
        dD_E = (self.theta_E * self.psi_E * E) - (1 / self.T_inc_D) * D_E# D_E
        dD_I = (self.theta_I * self.psi_I * I) + (1 / self.T_inc_D) * D_E - (1 / self.T_inf_D) * D_I # D_I 
        dR_mild = (1/self.T_inf)*(self.P_mild*I) + (1/self.T_inf_D)*(self.P_mild_D*D_I) - R_mild/self.T_recov_mild  # R_mild
        dR_severe_home = (1/self.T_inf)*(self.P_severe*I) + (1/self.T_inf_D)*(self.P_severe_D*D_I) - R_severe_home/self.T_hosp  # R_severe_home
        dR_severe_hosp = R_severe_home/self.T_hosp - R_severe_hosp/self.T_recov_severe# R_severe_hosp
        dR_fatal = (1/self.T_inf)*(self.P_fatal*I) + (1/self.T_inf_D)*(self.P_fatal_D*D_I) - R_fatal/self.T_death # R_fatal
        dC = R_mild/self.T_recov_mild + R_severe_hosp/self.T_recov_severe # C
        dD = R_fatal/self.T_death # D

        return [dS, dE, dI, dD_E, dD_I, dR_mild, dR_severe_home, dR_severe_hosp, dR_fatal, dC, dD]
    
    def solve_ode(self, total_no_of_days=200, time_step=1, method='Radau'):
        num_states = 11
        num_params = 7
        num_steps = 40
        num_train_steps = 7
        burn_in = 10
        mcmc_steps = 20
        observed = df_train['total_infected'][-num_train_steps:]
        num_train = len(df_train)


        sir_model = DifferentialEquation(
            func=SEIR_Test_obj.get_derivative,
            times=np.arange(0, num_steps, 1),
            n_states= num_states,
            n_theta= num_params,
            t0 = 0
        )

        with pm.Model() as model:
            R0 = pm.Uniform("R0", lower = 1, upper = 3)#(1.6, 3)
            T_inc = pm.Uniform("T_inc", lower = 1, upper = 5)#(3, 4)
            T_inf = pm.Uniform("T_inf", lower = 1, upper = 4)#(3, 4)
            T_recov_severe = pm.Uniform("T_recov_severe ", lower = 9, upper = 20)
            P_severe = pm.Uniform("P_severe", lower = 0.3, upper = 0.99)
            P_fatal = pm.Uniform("P_fatal", lower = 1e-6, upper = 0.3)
            intervention_amount = pm.Uniform("intervention_amount", lower = 0.3, upper = 1)

            ode_solution = sir_model(y0=init_vals , theta=[R0, T_inc, T_inf, T_recov_severe, P_severe,
                                                           P_fatal, intervention_amount])
            # The ode_solution has a shape of (n_times, n_states)

            predictions = ode_solution[num_train-num_train_steps-1:num_train-1]
            hospitalised = predictions[:, 6] + predictions[:, 7] + predictions[:, 8]
            recovered = predictions[:, 9]
            deceased = predictions[:, 10]
            total_infected = hospitalised + recovered + deceased
            total_infected = total_infected * num_patients 
            # sigma = pm.HalfNormal('sigma',
            #                      sigma=observed.std(),
            #                      shape=num_params)
            Y = pm.Normal('Y', mu = total_infected, observed=observed)

            prior = pm.sample_prior_predictive()
            trace = pm.sample(mcmc_steps, tune=burn_in , target_accept=0.9, cores=4)
            posterior_predictive = pm.sample_posterior_predictive(trace)

        return trace
        ##############
        #t_start = 0
        #t_final = total_no_of_days

        #time_steps = np.arange(t_start, total_no_of_days + time_step, time_step)
        
        #state_init_values_arr = [self.state_init_values[x] for x in self.state_init_values]

        #sol = solve_ivp(self.get_derivative, [t_start, t_final], 
        #                state_init_values_arr, method=method, t_eval=time_steps)

        #return sol
    def return_predictions(self, trace):
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
   

