import copy
from abc import abstractmethod
from collections import OrderedDict

from models.seir.compartmental_base import CompartmentalBase


class SEIR(CompartmentalBase):

    @abstractmethod
    def __init__(self, lockdown_R0=2.2, T_inf=2.9, T_inc=5.2, N=7e6,
                 starting_date='2020-03-09', observed_values=None,
                 E_tot_ratio=0.5, I_tot_ratio=0.5, **kwargs):

        STATES = ['S', 'E', 'I', 'R']
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
            'starting_date': starting_date,
            # Datetime value that corresponds to Day 0 of modelling
            'N': N,

            # Initialisation Params
            'E_hosp_ratio': E_tot_ratio,
            # Ratio for Exposed to hospitalised for initialisation
            'I_hosp_ratio': I_tot_ratio
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

        state_init_values['E'] = self.E_hosp_ratio * observed_values['total']
        state_init_values['I'] = self.I_hosp_ratio * observed_values['total']
        state_init_values['R'] = observed_values['total']

        nonSsum = sum(state_init_values.values())
        state_init_values['S'] = (self.N - nonSsum)
        for key in state_init_values.keys():
            state_init_values[key] = state_init_values[key] / self.N

        self.state_init_values = state_init_values

    @abstractmethod
    def get_derivative(self, t, y):
        pass

    @abstractmethod
    def predict(self, total_days, time_step, method):
        df_prediction = super().predict(total_days=total_days, time_step=time_step,
                                        method=method)

        df_prediction['active'] = float('nan')
        df_prediction['recovered'] = float('nan')
        df_prediction['deceased'] = float('nan')
        df_prediction['total'] = df_prediction['R']
