from collections import OrderedDict

import numpy as np
import copy

from models.seir.compartmental_base import CompartmentalBase


class SIR(CompartmentalBase):

    def __init__(self, lockdown_R0=2.2, T_inf=2.9, T_inc=5.2, N=7e6,
                 I_hosp_ratio=0.5, starting_date='2020-03-09', observed_values=None, **kwargs):
        """
        This class implements SIR.

        The state variables are :
        S : No of susceptible people
        I : No of infected people
        R : No of recovered people
        The sum total is is always N (total population)

        """

        STATES = ['S', 'I', 'R']
        R_STATES = [x for x in STATES if 'R_' in x]
        input_args = copy.deepcopy(locals())
        del input_args['self']
        del input_args['kwargs']
        p_params = {k: input_args[k] for k in input_args.keys() if 'P_' in k}
        t_params = {k: input_args[k] for k in input_args.keys() if 'T_' in k}

        params = {
            # R0 values
            'lockdown_R0': lockdown_R0,  # R0 value during lockdown

            # Transmission parameters
            'T_inc': T_inc,  # The incubation time of the infection
            'T_inf': T_inf,  # The duration for which an individual is infectious

            # Lockdown parameters
            'starting_date': starting_date,  # Datetime value that corresponds to Day 0 of modelling
            'N': N,

            # Initialisation Params
            # Ratio for Exposed to hospitalised for initialisation
            'I_hosp_ratio': I_hosp_ratio
            # Ratio for Infected to hospitalised for initialisation
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

        for state in R_STATES:
            statename = state.split('R_')[1]
            P_keyname = [k for k in p_params.keys() if k.split('P_')[1] == statename][0]
            state_init_values[state] = p_params[P_keyname] * observed_values['active']

        state_init_values['R'] = observed_values['total']
        state_init_values['I'] = I_hosp_ratio * observed_values['total']
        nonSsum = sum(state_init_values.values())

        state_init_values['S'] = (self.N - nonSsum)
        for key in state_init_values.keys():
            state_init_values[key] = state_init_values[key] / self.N

        self.state_init_values = state_init_values

    def get_derivative(self, t, y):
        """
        Calculates derivative at time t
        """

        # Init state variables
        for i, _ in enumerate(y):
            y[i] = max(y[i], 0)
        S, I, R = y

        self.T_trans = self.T_inf/self.lockdown_R0

        # Init derivative vector
        dydt = np.zeros(y.shape)

        # Write differential equations
        dydt[0] = - I * S / self.T_trans  # S
        dydt[1] = I * S / self.T_trans - (I / self.T_inf)  # I
        dydt[2] = I / self.T_inf  # R

        return dydt

    def predict(self, total_days=50, time_step=1, method='Radau'):
        """
        Returns predictions of the model
        """
        # Solve ODE get result
        df_prediction = super().predict(total_days=total_days,
                                        time_step=time_step, method=method)

        df_prediction['total'] = df_prediction['R']
        return df_prediction
