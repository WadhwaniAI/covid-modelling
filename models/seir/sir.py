import pandas as pd
import datetime

from abc import abstractmethod
from collections import OrderedDict

from utils.ode import ODE_Solver


class SIR:

    @abstractmethod
    def __init__(self, STATES, R_STATES, p_params, t_params, pre_lockdown_R0=3, lockdown_R0=2.2, post_lockdown_R0=None,
                 T_inf=2.9, T_inc=5.2, N=7e6, lockdown_day=10, lockdown_removal_day=75,
                 starting_date='2020-03-09', initialisation='intermediate', observed_values=None, ** kwargs):
        """
        This class implements SIR
        The model further implements
        - pre, post, and during lockdown behaviour
        - different initialisations : intermediate and starting

        The state variables are :

        S : No of susceptible people
        I : No of infected people
        R : No of recovered people

        The sum total is is always N (total population)

        """

        # If no value of post_lockdown R0 is provided, the model assumes the lockdown R0 post-lockdown
        if post_lockdown_R0 is None:
            post_lockdown_R0 = lockdown_R0

        params = {
            # R0 values
            'pre_lockdown_R0': pre_lockdown_R0,  # R0 value pre-lockdown
            'lockdown_R0': lockdown_R0,  # R0 value during lockdown
            'post_lockdown_R0': post_lockdown_R0,  # R0 value post-lockdown

            # Transmission parameters
            'T_inc': T_inc,  # The incubation time of the infection
            'T_inf': T_inf,  # The duration for which an individual is infectious

            # Lockdown parameters
            'starting_date': starting_date,  # Datetime value that corresponds to Day 0 of modelling
            'lockdown_day': lockdown_day,  # Number of days from the starting_date, after which lockdown is initiated
            'lockdown_removal_day': lockdown_removal_day,
            # Number of days from the starting_date, after which lockdown is removed
            'N': N,

        }

        for key in params:
            setattr(self, key, params[key])

        for key in p_params:
            setattr(self, key, p_params[key])

        for key in t_params:
            setattr(self, key, t_params[key])

        # Initialisation
        state_init_values = OrderedDict()
        for key in STATES:
            state_init_values[key] = 0
        if initialisation == 'starting':
            init_infected = max(observed_values['init_infected'], 1)
            state_init_values['S'] = (self.N - init_infected) / self.N
            state_init_values['I'] = init_infected / self.N

        if initialisation == 'intermediate':
            for state in R_STATES:
                statename = state.split('R_')[1]
                P_keyname = [k for k in p_params.keys() if k.split('P_')[1] == statename][0]
                state_init_values[state] = p_params[P_keyname] * observed_values['hospitalised']

            state_init_values['R'] = observed_values['recovered']
            state_init_values['D'] = observed_values['deceased']
            state_init_values['I'] = observed_values['hospitalised']

            nonSsum = sum(state_init_values.values())
            state_init_values['S'] = (self.N - nonSsum)
            for key in state_init_values.keys():
                state_init_values[key] = state_init_values[key] / self.N

        self.state_init_values = state_init_values

    @abstractmethod
    def get_derivative(self, t, y):
        """
        Calculates derivative at time t
        """
        pass

    @abstractmethod
    def predict(self, total_days=50, time_step=1, method='Radau'):
        """
        Returns predictions of the model
        """
        # Solve ODE get result
        solver = ODE_Solver()
        state_init_values_arr = [self.state_init_values[x]
                                 for x in self.state_init_values]

        sol = solver.solve_ode(state_init_values_arr, self.get_derivative, method=method,
                               total_days=total_days, time_step=time_step)

        states_time_matrix = (sol.y * self.N).astype('int')

        dataframe_dict = {}
        for i, key in enumerate(self.state_init_values.keys()):
            dataframe_dict[key] = states_time_matrix[i]

        df_prediction = pd.DataFrame.from_dict(dataframe_dict)
        df_prediction['date'] = pd.date_range(self.starting_date,
                                              self.starting_date + datetime.timedelta(days=df_prediction.shape[0] - 1))
        columns = list(df_prediction.columns)
        columns.remove('date')
        df_prediction = df_prediction[['date'] + columns]
        return df_prediction

