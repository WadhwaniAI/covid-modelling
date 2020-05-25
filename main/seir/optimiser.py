import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from hyperopt import hp, tpe, fmin, Trials
from tqdm import tqdm
from tqdm.notebook import tqdm

from collections import OrderedDict
import itertools
from functools import partial, reduce
import datetime
from joblib import Parallel, delayed

from models.seir.seir_testing import SEIR_Testing
from main.seir.losses import Loss_Calculator

class Optimiser():

    def __init__(self):
        self.loss_calculator = Loss_Calculator()

    def solve(self, variable_params, default_params, df_true, start_date=None, end_date=None, 
              state_init_values=None, initialisation='starting', loss_indices=[-20, -10]):
        params_dict = {**variable_params, **default_params}
        if end_date == None:
            end_date = df_true.iloc[-1, :]['date']
        else:
            if type(end_date) is str:
                end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_date != None:
            if type(start_date) is str:
                start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            params_dict['starting_date'] = start_date
        solver = SEIR_Testing(**params_dict)
        total_days = (end_date - params_dict['starting_date']).days
        sol = solver.solve_ode(total_no_of_days=total_days, time_step=1, method='Radau')
        df_prediction = solver.return_predictions(sol)
        return df_prediction


    # TODO add cross validation support
    def solve_and_compute_loss(self, variable_params, default_params, df_true, total_days, 
                               which_compartments=['hospitalised', 'recovered', 'total_infected', 'deceased'], 
                               loss_indices=[-20, -10], loss_method='rmse', return_dict=False, 
                               initialisation='starting'):
        params_dict = {**variable_params, **default_params}
        
        # Returning a very high loss value for the cases where the sampled values of P_severe and P_fatal are > 1
        if params_dict['P_severe'] + params_dict['P_fatal'] > 1:
            return 1e10

        solver = SEIR_Testing(**params_dict)
        sol = solver.solve_ode(total_no_of_days=total_days - 1, time_step=1, method='Radau')
        df_prediction = solver.return_predictions(sol)

        # Choose which indices to calculate loss on
        # TODO Add slicing capabilities on the basis of date
        if loss_indices == None:
            df_prediction_slice = df_prediction
            df_true_slice = df_true
        else:
            df_prediction_slice = df_prediction.iloc[loss_indices[0]:loss_indices[1], :]
            df_true_slice = df_true.iloc[loss_indices[0]:loss_indices[1], :]

        df_prediction_slice.reset_index(inplace=True, drop=True)
        df_true_slice.reset_index(inplace=True, drop=True)
        if return_dict:
            loss = self.loss_calculator.calc_loss_dict(df_prediction_slice, df_true_slice, method=loss_method)
        else:
            loss = self.loss_calculator.calc_loss(df_prediction_slice, df_true_slice, 
                                                  which_compartments=which_compartments, method=loss_method)
        return loss

    def _create_dict(self, param_names, values):
        params_dict = {param_names[i]: values[i] for i in range(len(values))}
        return params_dict

    def init_default_params(self, df_train, N=1e7, lockdown_date='2020-03-25', lockdown_removal_date='2020-05-31', 
                            T_hosp=0.001, initialisation='intermediate', train_period=7, start_date=None,
                            observed_values=None):

        intervention_date = datetime.datetime.strptime(lockdown_date, '%Y-%m-%d')
        lockdown_removal_date = datetime.datetime.strptime(lockdown_removal_date, '%Y-%m-%d')

        if initialisation == 'intermediate':
            observed_values = df_train.iloc[-train_period, :]
            start_date = observed_values['date']
        if initialisation == 'starting':
            raise AssertionError

        default_params = {
            'N' : N,
            'lockdown_day' : (intervention_date - start_date).days,
            'lockdown_removal_day': (lockdown_removal_date - start_date).days,
            'T_hosp' : T_hosp,
            'starting_date' : start_date,
            'observed_values': observed_values
        }

        return default_params

    def gridsearch(self, df_true, default_params, variable_param_ranges, method='rmse', loss_indices=[-20, -10], 
                   which_compartments=['total_infected'], debug=False):
        """
        What variable_param_ranges is supposed to look like
        variable_param_ranges = {
            'R0' : np.linspace(1.8, 3, 13),
            'T_inc' : np.linspace(3, 5, 5),
            'T_inf' : np.linspace(2.5, 3.5, 5),
            'T_recov_severe' : np.linspace(11, 15, 10),
            'P_severe' : np.linspace(0.3, 0.9, 25),
            'intervention_amount' : np.linspace(0.4, 1, 31)
        }
        """
        total_days = len(df_true['date'])

        rangelists = list(variable_param_ranges.values())
        cartesian_product_tuples = itertools.product(*rangelists)
        list_of_param_dicts = [self._create_dict(list(variable_param_ranges.keys()), values) for values in cartesian_product_tuples]

        partial_solve_and_compute_loss = partial(self.solve_and_compute_loss, default_params=default_params, df_true=df_true,
                                                 total_days=total_days, loss_method=method, loss_indices=loss_indices, 
                                                 which_compartments=which_compartments)
        
        # If debugging is enabled the gridsearch is not parallelised
        if debug:
            loss_array = []
            for params_dict in tqdm(list_of_param_dicts):
                loss_array.append(partial_solve_and_compute_loss(params_dict))
        else:
            loss_array = Parallel(n_jobs=40)(delayed(partial_solve_and_compute_loss)(params_dict) for params_dict in tqdm(list_of_param_dicts))
                    
        return loss_array, list_of_param_dicts

    def bayes_opt(self, df_true, default_params, variable_param_ranges, total_days=None, method='rmse', num_evals=3500, 
                  loss_indices=[-20, -10], which_compartments=['total_infected'], initialisation='starting'):
        """
        What variable_param_ramges is supposed to look like : 
        variable_param_ranges = {
            'R0' : hp.uniform('R0', 1.6, 3),
            'T_inc' : hp.uniform('T_inc', 4, 5),
            'T_inf' : hp.uniform('T_inf', 3, 4),
            'T_recov_severe' : hp.uniform('T_recov_severe', 9, 20),
            'P_severe' : hp.uniform('P_severe', 0.3, 0.99),
            'intervention_amount' : hp.uniform('intervention_amount', 0.3, 1)
        }
        """
        if total_days == None:
            total_days = len(df_true['date'])
        
        partial_solve_and_compute_loss = partial(self.solve_and_compute_loss, default_params=default_params, df_true=df_true,
                                                 total_days=total_days, loss_method=method, loss_indices=loss_indices, 
                                                 which_compartments=which_compartments, initialisation=initialisation)
        
        searchspace = variable_param_ranges
        
        trials = Trials()
        best = fmin(partial_solve_and_compute_loss,
                    space=searchspace,
                    algo=tpe.suggest,
                    max_evals=num_evals,
                    trials=trials)
        
        return best, trials