import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pdb

class SEIR:
  def __init__(self, params, state_init_values):
    self.params = params
    self.state_init_values = state_init_values
    self.time_params = self.params[:-4]
    self.N = self.params[-4]
    self.intervention_day = self.params[-3]
    self.intervention_choice = self.params[-1]
    self.intervention_duration = self.params[-2]

  def get_impact(self, choice):
    return(1+2*choice)


  def get_derivative(self, t, y):
    # Init state variables
    S, I, R = y

    # Init time parameters and probabilities
    T_trans, T_treat = self.time_params
    
    # Modelling the intervention
    for i in range(len(self.intervention_day)):
        if (t >= self.intervention_day[i] and t<self.intervention_day[i]+self.intervention_duration[i]):
          T_trans = self.get_impact(self.intervention_choice[i])*T_trans

    # Init derivative vector
    dydt = np.zeros(y.shape)
    # Write differential equations
    dydt[0] = -I*S/(T_trans)
    dydt[1] = I*S/(T_trans) - I/T_treat
    dydt[2] = I/T_treat

    return dydt

  def solve_ode(self, total_no_of_days=200, time_step=2):
    t_start = 0
    t_final = total_no_of_days
    time_steps = np.arange(t_start, total_no_of_days + time_step, time_step)

    sol = solve_ivp(self.get_derivative, [t_start, t_final], 
                    self.state_init_values, method='RK45', t_eval=time_steps)
    
    return sol
