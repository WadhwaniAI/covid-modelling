import numpy as np
import copy

from models.seir.sir_base import SIRBase


class SIR(SIRBase):

    def __init__(self, lockdown_R0=2.2, T_inf=2.9, T_inc=5.2, N=7e6,
                 I_hosp_ratio=0.5, I_tot_ratio=0.5, starting_date='2020-03-09', observed_values=None):
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

        STATES = ['S', 'I', 'R']
        R_STATES = [x for x in STATES if 'R_' in x]
        input_args = copy.deepcopy(locals())
        del input_args['self']
        del input_args['kwargs']
        p_params = {k: input_args[k] for k in input_args.keys() if 'P_' in k}
        t_params = {k: input_args[k] for k in input_args.keys() if 'T_' in k}
        input_args['p_params'] = p_params
        input_args['t_params'] = t_params
        super().__init__(**input_args)

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

        df_prediction['hospitalised'] = float('nan')
        df_prediction['recovered'] = float('nan')
        df_prediction['deceased'] = float('nan')
        df_prediction['total_infected'] = df_prediction['R']
        return df_prediction
