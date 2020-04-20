import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pdb

class SEIR:
  def __init__(self, params, state_init_values):
    self.params = params
    self.state_init_values = state_init_values
    self.time_params = self.params[:-7]
    self.p_params = self.params[-7:-4]
    self.N = self.params[-4]
    self.intervention_day = self.params[-3]
    # self.intervention_amount = self.params[-3]
    self.intervention_duration = self.params[-2]
    self.intervention_choice = self.params[-1]

  def get_impact(self, choice):
    return(1+1*choice)

  def get_derivative(self, t, y):
    # Init state variables
    S, E, I, R_mild, R_severe, R_severe_hosp, R_fatal, C, D = y

    # Init time parameters and probabilities
    T_trans, T_inc, T_inf, T_recov_mild, T_hosp, T_recov_severe, T_death = self.time_params
    P_mild, P_severe, P_fatal = self.p_params
    
    # Modelling the intervention
    for i in range(len(self.intervention_day)):
        if (t >= self.intervention_day[i] and t<self.intervention_day[i]+self.intervention_duration[i]):
          T_trans = self.get_impact(self.intervention_choice[i])*T_trans

    # Init derivative vector
    dydt = np.zeros(y.shape)
    # Write differential equations
    dydt[0] = -I*S/(T_trans)
    dydt[1] = I*S/(T_trans) - E/T_inc
    dydt[2] = E/T_inc - I/T_inf
    dydt[3] = (P_mild*I)/T_inf - R_mild/T_recov_mild
    dydt[4] = (P_severe*I)/T_inf - R_severe/T_hosp
    dydt[5] = R_severe/T_hosp - R_severe_hosp/T_recov_severe
    dydt[6] = (P_fatal*I)/T_inf - R_fatal/T_death
    dydt[7] = R_mild/T_recov_mild + R_severe_hosp/T_recov_severe
    dydt[8] = R_fatal/T_death

    return dydt

  def solve_ode(self, total_no_of_days=200, time_step=2):
    t_start = 0
    t_final = total_no_of_days
    time_steps = np.arange(t_start, total_no_of_days + time_step, time_step)

    sol = solve_ivp(self.get_derivative, [t_start, t_final], 
                    self.state_init_values, method='RK45', t_eval=time_steps)
    
    return sol
