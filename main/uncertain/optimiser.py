import itertools as it
import collections as cl

import pandas as pd
from scipy import constants

from models.seir.seir_testing import SEIR_Testing

Lockdown = cl.namedtuple('Lockdown', 'start, end')

def offset(df, dates):
    zero = df.index.min()
    for i in dates:
        diff = i - zero
        yield diff.total_seconds() / constants.day

class Optimiser:
    @staticmethod
    def dict2str(d):
        return ', '.join([ ': '.join(map(str, x)) for x in d.items() ])

    def __init__(self,
                 infected,
                 N=1e7,
                 lockdown=None,
                 T_hosp=0.001,
                 P_fatal=0.01):
        self.infected = infected

        if lockdown is None:
            drange = ('2020-03-25', '2020-05-03')
            lockdown = Lockdown(*map(pd.to_datetime, drange))
        (day, removal) = offset(self.infected, lockdown)

        self.defaults = {
            'N': N,
            'init_infected': self.infected.iloc[0].to_numpy().item(),
            'intervention_day': day,
            'intervention_removal_day': removal,
            'T_hosp': T_hosp,
            'P_fatal': P_fatal,
            'starting_date': self.infected.index.min(),
        }

    def __repr__(self):
        return self.dict2str(self.defaults)

    def solve(self, parameters, infected=None, end=None):
        if infected is None:
            infected = self.infected

        kwargs = dict(self.defaults)
        assert not any(x in parameters for x in kwargs)
        kwargs.update(parameters)

        if end is None:
            end = infected.index.max()
        duration = end - self.defaults['starting_date']
        days = duration.total_seconds() / constants.day

        solver = SEIR_Testing(**kwargs)
        solution = solver.solve_ode(total_no_of_days=days,
                                    time_step=1,
                                    method='Radau')
        df = (solver
              .return_predictions(solution)
              .set_index('date'))

        return df
