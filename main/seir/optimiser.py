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
from functools import partial
import datetime
from joblib import Parallel, delayed

from models.seir.seir_testing import SEIR_Testing
from main.seir.losses import Loss_Calculator

class Optimiser():

    def __init__(self):
        self.loss_calculator = Loss_Calculator()

    def solve(self, variable_params, default_params, df_true, end_date=None):
        params_dict = {**variable_params, **default_params}
        solver = SEIR_Testing(**params_dict)
        if end_date == None:
            end_date = df_true.iloc[-1, :]['date']
        else:
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
#         total_days = (params_dict['starting_date'] - end_date).days,
        total_days = len(df_true['date'])
        sol = solver.solve_ode(total_no_of_days=total_days - 1, time_step=1, method='Radau')
        df_prediction = solver.return_predictions(sol)
        return df_prediction


    # TODO add cross validation support
    def solve_and_compute_loss(self, variable_params, default_params, df_true, total_days, 
                               which_compartments=['hospitalised', 'recovered', 'total', 'fatalities'], 
                               loss_indices=[-20, -10], loss_method='rmse', return_dict=False):
        params_dict = {**variable_params, **default_params}
        solver = SEIR_Testing(**params_dict)
        sol = solver.solve_ode(total_no_of_days=total_days - 1, time_step=1, method='Radau')
        df_prediction = solver.return_predictions(sol)

        # Choose which indices to calculate loss on
        # TODO Add slicing capabilities on the basis of date
        if loss_indices == None:
            df_prediction_slice = df_prediction.iloc[:, :]
            df_true_slice = df_true.iloc[:, :]
        else:
            df_prediction_slice = df_prediction.iloc[loss_indices[0]:loss_indices[1], :]
            df_true_slice = df_true.iloc[loss_indices[0]:loss_indices[1], :]
        if return_dict:
            loss = self.loss_calculator.calc_loss_dict(df_prediction_slice, df_true_slice, method=loss_method)
        else:
            loss = self.loss_calculator.calc_loss(df_prediction_slice, df_true_slice, 
                                                  which_compartments=which_compartments, method=loss_method)
        return loss

    def _create_dict(self, param_names, values):
        params_dict = {param_names[i]: values[i] for i in range(len(values))}
        return params_dict

    def init_default_params(self, df_true, N=1e7, lockdown_date='2020-03-25', lockdown_removal_date='2020-05-03', 
                            T_hosp=0.001, P_fatal=0.01):
        init_infected = max(df_true.iloc[0, :]['total_infected'], 1)
        start_date = df_true.iloc[0, :]['date']
        intervention_date = datetime.datetime.strptime(lockdown_date, '%Y-%m-%d')
        lockdown_removal_date = datetime.datetime.strptime(lockdown_removal_date, '%Y-%m-%d')

        default_params = {
            'N' : N,
            'init_infected' : init_infected,
            'intervention_day' : (intervention_date - start_date).days,
            'intervention_removal_day': (lockdown_removal_date - start_date).days,
            'T_hosp' : T_hosp,
            'P_fatal' : P_fatal,
            'starting_date' : start_date
        }
        return default_params

    def gridsearch(self, df_true, default_params, variable_param_ranges, method='rmse', loss_indices=[-20, -10], debug=False):
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
                                                 total_days=total_days, loss_method=method, loss_indices=loss_indices)
        
        # If debugging is enabled the gridsearch is not parallelised
        if debug:
            loss_array = []
            for params_dict in tqdm(list_of_param_dicts):
                loss_array.append(partial_solve_and_compute_loss(params_dict))
        else:
            loss_array = Parallel(n_jobs=40)(delayed(partial_solve_and_compute_loss)(params_dict) for params_dict in tqdm(list_of_param_dicts))
                    
        return loss_array, list_of_param_dicts

    def bayes_opt(self, df_true, default_params, variable_param_ranges, method='rmse', num_evals=3500, loss_indices=[-20, -10]):
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
        total_days = len(df_true['date'])
        
        partial_solve_and_compute_loss = partial(self.solve_and_compute_loss, default_params=default_params, df_true=df_true,
                                                 total_days=total_days, loss_method=method, loss_indices=loss_indices)
        
        searchspace = variable_param_ranges
        
        trials = Trials()
        best = fmin(partial_solve_and_compute_loss,
                    space=searchspace,
                    algo=tpe.suggest,
                    max_evals=num_evals,
                    trials=trials)
        
        return best, trials

    def evaluate_losses(self, best, default_params, df_train, df_val):
        start_date = df_train.iloc[0, 0]
        simulate_till = df_val.iloc[-1, 0]
        total_no_of_days = (simulate_till - start_date).days + 1
        no_of_train_days = (df_train.iloc[-1, 0] - start_date).days + 1
        no_of_val_days = total_no_of_days - no_of_train_days
        
        final_params = {**best, **default_params}
        vanilla_params, testing_params, state_init_values = init_params(**final_params)
        solver = SEIR_Testing(vanilla_params, testing_params, state_init_values)
        sol = solver.solve_ode(total_no_of_days=total_no_of_days - 1, time_step=1, method='Radau')
        states_time_matrix = (sol.y*vanilla_params['N']).astype('int')

        train_output = states_time_matrix[:, :no_of_train_days]
        val_output = states_time_matrix[:, -no_of_val_days:]
        
        rmse_loss = self.loss_calculator.calc_loss_dict(
            train_output, df_train, method='rmse')
        rmse_loss = pd.DataFrame.from_dict(rmse_loss, orient='index', columns=['rmse'])
        
        mape_loss = self.loss_calculator.calc_loss_dict(
            train_output, df_train, method='mape')
        mape_loss = pd.DataFrame.from_dict(mape_loss, orient='index', columns=['mape'])
        
        train_losses = pd.concat([rmse_loss, mape_loss], axis=1)
        
        pred_hospitalisations = val_output[6] + val_output[7] + val_output[8]
        pred_recoveries = val_output[9]
        pred_fatalities = val_output[10]
        pred_infectious_unknown = val_output[2] + val_output[4]
        pred_total_cases = pred_hospitalisations + pred_recoveries + pred_fatalities
        print('Pred', pred_hospitalisations, pred_total_cases)
        print('True', df_val['hospitalised'].to_numpy(), df_val['total_infected'].to_numpy())
        
        rmse_loss = self.loss_calculator.calc_loss_dict(
            val_output, df_val, method='rmse')
        rmse_loss = pd.DataFrame.from_dict(rmse_loss, orient='index', columns=['rmse'])
        
        mape_loss = self.loss_calculator.calc_loss_dict(
            val_output, df_val, method='mape')
        mape_loss = pd.DataFrame.from_dict(mape_loss, orient='index', columns=['mape'])
        
        val_losses = pd.concat([rmse_loss, mape_loss], axis=1)
        
        print(train_losses)
        print(val_losses)
        
        return train_losses, val_losses
