import numpy as np
import copy

from models.seir.seir_base import SEIRBase

class SEIRHD(SEIRBase):
    def __init__(self, lockdown_R0=2.2, T_inf=2.9, T_inc=5.2, P_fatal=0.02, T_recov_fatal=32, T_recov=14, N=7e6,
                 starting_date='2020-03-09', observed_values=None, E_hosp_ratio=0.5, I_hosp_ratio=0.5, **kwargs):
        """
        This class implements SEIR + Hospitalisation

        The state variables are : 

        S : No of susceptible people
        E : No of exposed people
        I : No of infected people
        R_recov : No of people recovering from a curable version of the infection
        R_fatal : No of people recovering from a fatal version of the infection
        C : No of recovered people
        D : No of deceased people 

        The sum total is is always N (total population)

        """

        """
        The parameters are : 

        R0 values - 
        lockdown_R0: R0 value during lockdown (float)

        Transmission parameters - 
        T_inc: The incubation time of the infection (float)
        T_inf: The duration for which an individual is infectious (float)

        Probability of contracting different types of infections - 
        P_recov: Probability of contracting a severe infection (float - [0, 1])
        P_fatal: Probability of contracting a fatal infection (float - [0, 1])

        Clinical time parameters - 
        T_recov: Time it takes for an individual with a severe infection to recover (float)
        T_recov_fatal: Time it takes for an individual with a fatal infection to die (float)

        Lockdown parameters - 
        starting_date: Datetime value that corresponds to Day 0 of modelling (datetime/str)

        Misc - 
        N: Total population
        """
        STATES = ['S', 'E', 'I', 'R_recov', 'R_fatal', 'C', 'D']
        R_STATES = [x for x in STATES if 'R_' in x]
        input_args = copy.deepcopy(locals())
        del input_args['self']
        del input_args['kwargs']
        p_params = {k: input_args[k] for k in input_args.keys() if 'P_' in k}
        t_params = {k: input_args[k] for k in input_args.keys() if 'T_recov' in k}
        p_params['P_recov'] = 1 - p_params['P_fatal']
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
        S, E, I, R_recov, R_fatal, C, D = y

        self.T_trans = self.T_inf/self.lockdown_R0

        # Init derivative vector
        dydt = np.zeros(y.shape)

        # Write differential equations
        dydt[0] = - I * S / self.T_trans  # S
        dydt[1] = I * S / self.T_trans - (E / self.T_inc)  # E
        dydt[2] = E / self.T_inc - I / self.T_inf  # I
        dydt[3] = (1/self.T_inf)*(self.P_recov*I) - R_recov/self.T_recov #R_recov
        dydt[4] = (1/self.T_inf)*(self.P_fatal*I) - R_fatal/self.T_recov_fatal # R_fatal
        dydt[5] = R_recov/self.T_recov  # C
        dydt[6] = R_fatal/self.T_recov_fatal # D

        return dydt

    def predict(self, total_days=50, time_step=1, method='Radau'):
        """
        Returns predictions of the model
        """
        # Solve ODE get result
        df_prediction = super().predict(total_days=total_days,
                                        time_step=time_step, method=method)

        df_prediction['active'] = df_prediction['R_recov'] + df_prediction['R_fatal']
        df_prediction['recovered'] = df_prediction['C']
        df_prediction['deceased'] = df_prediction['D']
        df_prediction['total'] = df_prediction['active'] + \
            df_prediction['recovered'] + df_prediction['deceased']
        return df_prediction
