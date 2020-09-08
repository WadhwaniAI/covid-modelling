import numpy as np
from scipy.integrate import solve_ivp


class ODE_Solver():

    def __init__(self):
        pass

    def solve_ode(self, state_init_values, func, total_days=200, time_step=1, method='Radau'):
        """
        Solves ODE
        """
        t_start = 0
        t_final = total_days
        time_steps = np.arange(t_start, total_days + time_step, time_step)
        
        sol = solve_ivp(func, [t_start, t_final],
                        state_init_values, method=method, t_eval=time_steps)

        return sol
