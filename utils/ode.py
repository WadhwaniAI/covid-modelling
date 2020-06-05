


class ODE_Solver():

    def __init__(self):
        pass

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