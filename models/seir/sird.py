import numpy as np
import copy

from models.seir.sir_base import SIRBase


class SIRD(SIRBase):

    def __init__(self, pre_lockdown_R0=3, lockdown_R0=2.2, post_lockdown_R0=None, T_inf=2.9, T_inc=5.2, T_fatal=7,
                 N=7e6, lockdown_day=10, lockdown_removal_day=75, starting_date='2020-03-09',
                 initialisation='intermediate', observed_values=None, E_hosp_ratio=0.5, I_hosp_ratio=0.5, ** kwargs):
        """
        This class implements SIR
        The model further implements
        - pre, post, and during lockdown behaviour
        - different initialisations : intermediate and starting

        The state variables are :

        S : No of susceptible people
        I : No of infected people
        R : No of recovered people
        D: No of deceased people

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

        STATES = ['S', 'I', 'R', 'D']
        R_STATES = [x for x in STATES if 'R_' in x]
        input_args = copy.deepcopy(locals())
        del input_args['self']
        del input_args['kwargs']
        p_params = {k: input_args[k] for k in input_args.keys() if 'P_' in k}
        t_params = {k: input_args[k] for k in input_args.keys() if 'T_' in k}
        input_args['p_params'] = p_params
        input_args['t_params'] = t_params
        super().__init__(**input_args)
        if initialisation == 'intermediate':
            self.state_init_values['D'] = observed_values['deceased']

    def get_derivative(self, t, y):
        """
        Calculates derivative at time t
        """

        # Init state variables
        for i, _ in enumerate(y):
            y[i] = max(y[i], 0)
        S, I, R, D = y

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
        dydt[0] = - I * S / self.T_trans  # S
        dydt[1] = I * S / self.T_trans - (I / self.T_inf) - (I / self.T_fatal)  # I
        dydt[2] = I / self.T_inf  # R
        dydt[3] = I * self.T_fatal  # D

        return dydt

    def predict(self, total_days=50, time_step=1, method='Radau'):
        """
        Returns predictions of the model
        """
        # Solve ODE get result
        df_prediction = super().predict(total_days=total_days,
                                        time_step=time_step, method=method)

        df_prediction['hospitalised'] = float('nan')
        df_prediction['recovered'] = float('nan')
        df_prediction['deceased'] = df_prediction['D']
        df_prediction['total_infected'] = df_prediction['R']
        return df_prediction
