from abc import ABC, abstractmethod
from models import Model

import pandas as pd
import numpy as np
import datetime

from utils.ode import ODE_Solver

class SEIR(Model):

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def get_derivative(self, t, y):
        pass

    @abstractmethod
    def predict(self, total_days, time_step, method):
        # Solve ODE get result
        solver = ODE_Solver()
        state_init_values_arr = [self.state_init_values[x]
                                 for x in self.state_init_values]

        sol = solver.solve_ode(state_init_values_arr, self.get_derivative, method=method, 
                               total_days=total_days, time_step=time_step)

        states_time_matrix = (sol.y*self.N).astype('int')

        dataframe_dict = {}
        for i, key in enumerate(self.state_init_values.keys()):
            dataframe_dict[key] = states_time_matrix[i]
        
        df_prediction = pd.DataFrame.from_dict(dataframe_dict)
        df_prediction['date'] = pd.date_range(self.starting_date, self.starting_date + datetime.timedelta(days=df_prediction.shape[0] - 1))
        columns = list(df_prediction.columns)
        columns.remove('date')
        df_prediction = df_prediction[['date'] + columns]
        return df_prediction
