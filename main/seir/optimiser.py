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
    """Class which implements all optimisation related activites (training, evaluation, etc)
    """

    def __init__(self):
        self.loss_calculator = Loss_Calculator()

    def solve(self, variable_params : dict, default_params :dict, df_true : pd.DataFrame, model=SEIR_Testing, 
              start_date=None, end_date=None):
        """This function solves the ODE for an input of params (but does not compute loss)

        Arguments:
            variable_params {dict} -- The values for the params that are variable across the searchspace
            default_params {dict} -- The values for the params that are fixed across the searchspace
            df_true {pd.DataFrame} -- The training dataset

        Keyword Arguments:
            model {class} -- The epi model class to be used for modelling (default: {SEIR_Testing})
            start_date {str} -- The start date (usually not specifed, inferred from the params_dict) (default: {None})
            end_date {str} -- Last date of projection (default: {None})

        Returns:
            pd.DataFrame -- DataFrame of predictions
        """

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
        solver = model(**params_dict)
        total_days = (end_date - params_dict['starting_date']).days
        df_prediction = solver.predict(total_days=total_days)
        # df_prediction = solver.predict()
        return df_prediction


    # TODO add cross validation support
    def solve_and_compute_loss(self, variable_params, default_params, df_true, total_days, model=SEIR_Testing,
                               which_compartments=['hospitalised', 'recovered', 'total_infected', 'deceased'], 
                               loss_indices=[-20, -10], loss_method='rmse', return_dict=False):
        """The function that computes solves the ODE for a given set of input params and computes loss on train set

        Arguments:
            variable_params {dict} -- The values for the params that are variable across the searchspace
            default_params {dict} -- The values of the params that are fixed across the searchspace
            df_true {pd.DataFrame} -- The train set
            total_days {int} -- Total number of days into the future for which we want to simulate

        Keyword Arguments:
            model {class} -- The epi model class to be used for modelling (default: {SEIR_Testing})
            which_compartments {list} -- Which compartments to apply loss on 
            (default: {['hospitalised', 'recovered', 'total_infected', 'deceased']})
            loss_indices {list} -- Which indices of the train set to apply loss on (default: {[-20, -10]})
            loss_method {str} -- Loss Method (default: {'rmse'})
            return_dict {bool} -- If True, instead of returning single loss value, will return loss value 
            for every compartment (default: {False})

        Returns:
            float -- The loss value
        """
        params_dict = {**variable_params, **default_params}
        
        # Returning a very high loss value for the cases where the sampled values of P_severe and P_fatal are > 1
        # if params_dict['P_severe'] + params_dict['P_fatal'] > 1:
        #     return 1e10

        solver = model(**params_dict)
        # solver.solve_ode(total_no_of_days=total_days - 1, time_step=1, method='Radau')
        df_prediction = solver.predict(total_days=total_days)

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
        """Helper function for creating dict of all parameters

        Arguments:
            param_names {arr} -- Names of the parameters
            values {arr} -- Their corresponding values

        Returns:
            dict -- Dict of all parameters
        """
        params_dict = {param_names[i]: values[i] for i in range(len(values))}
        return params_dict

    def init_default_params(self, df_train, N=1e7, lockdown_date='2020-03-25', lockdown_removal_date='2020-06-30', 
                            initialisation='intermediate', train_period=7, start_date=None,
                            observed_values=None):
        """Function for creating all default params for the optimisation (hyperopt/gridsearch)

        Arguments:
            df_train {pd.DataFramw} -- The train dataset

        Keyword Arguments:
            N {float} -- Population of region (default: {1e7})
            lockdown_date {str} -- The date on which lockdown is implemented (default: {'2020-03-25'})
            lockdown_removal_date {str} -- The date on which lockdown is removed (default: {'2020-05-31'})
            initialisation {str} -- The method of initialisation (default: {'intermediate'})
            train_period {int} -- The number of days for which the model is trained (default: {7})
            start_date {str} -- If initialisation=='starting', start_date must be provided (default: {None})
            observed_values {pd.Series} -- This is a row of a pandas dataframe that corresponds to the observed values 
            on the initialisation date (to initialise the latent params) (default: {None})

        Returns:
            dict -- Dict of default params
        """

        intervention_date = datetime.datetime.strptime(lockdown_date, '%Y-%m-%d')
        lockdown_removal_date = datetime.datetime.strptime(lockdown_removal_date, '%Y-%m-%d')

        if initialisation == 'intermediate':
            observed_values = df_train.iloc[-train_period, :]
            start_date = observed_values['date']
        if initialisation == 'starting':
            assert start_date != None

        default_params = {
            'N' : N,
            'lockdown_day' : (intervention_date - start_date).days,
            'lockdown_removal_day': (lockdown_removal_date - start_date).days,
            'starting_date' : start_date,
            'observed_values': observed_values
        }

        return default_params

    def gridsearch(self, df_true, default_params, variable_param_ranges, model=SEIR_Testing, method='rmse',
                   loss_indices=[-20, -10], which_compartments=['total_infected'], debug=False):
        """Implements gridsearch based optimisation

        Arguments:
            df_true {pd.DataFrame} -- The train set
            default_params {dict} -- Dict of default (fixed) params
            variable_param_ranges {dict} -- The range of variable params (the searchspace)

        An example of variable_param_ranges :
        variable_param_ranges = {
            'R0' : np.linspace(1.8, 3, 13),
            'T_inc' : np.linspace(3, 5, 5),
            'T_inf' : np.linspace(2.5, 3.5, 5),
            'T_recov_severe' : np.linspace(11, 15, 10),
            'P_severe' : np.linspace(0.3, 0.9, 25),
            'intervention_amount' : np.linspace(0.4, 1, 31)
        }

        Keyword Arguments:
            model {class} -- The epi model class to be used for modelling (default: {SEIR_Testing})
            method {str} -- The loss method (default: {'rmse'})
            loss_indices {list} -- The indices on the train set to apply the loss on (default: {[-20, -10]})
            which_compartments {list} -- Which compartments to apply loss on (default: {['total_infected']})
            debug {bool} -- If debug is true, gridsearch is not parellelised. For debugging (default: {False})

        Returns:
            arr, list(dict) -- Array of loss values, and a list of parameter dicts
        """
        total_days = len(df_true['date'])

        rangelists = list(variable_param_ranges.values())
        cartesian_product_tuples = itertools.product(*rangelists)
        list_of_param_dicts = [self._create_dict(list(
            variable_param_ranges.keys()), values) for values in cartesian_product_tuples]

        partial_solve_and_compute_loss = partial(self.solve_and_compute_loss, model=model, 
                                                 default_params=default_params, total_days=total_days, 
                                                 loss_method=method, loss_indices=loss_indices, df_true=df_true,
                                                 which_compartments=which_compartments)
        
        # If debugging is enabled the gridsearch is not parallelised
        if debug:
            loss_array = []
            for params_dict in tqdm(list_of_param_dicts):
                loss_array.append(partial_solve_and_compute_loss(params_dict))
        else:
            loss_array = Parallel(n_jobs=40)(delayed(partial_solve_and_compute_loss)(params_dict) for params_dict in tqdm(list_of_param_dicts))
                    
        return loss_array, list_of_param_dicts

    def bayes_opt(self, df_true, default_params, variable_param_ranges, model=SEIR_Testing, total_days=None, 
                  method='rmse', num_evals=3500, loss_indices=[-20, -10], which_compartments=['total_infected']):
        """Implements Bayesian Optimisation using hyperopt library

        Arguments:
            df_true {pd.DataFrame} -- The training dataset
            default_params {str} -- Dict of default (static) params
            variable_param_ranges {dict} -- The ranges for the variable params (the searchspace)

        An example of variable_param_ranges : 
        variable_param_ranges = {
            'lockdown_R0' : hp.uniform('R0', 0, 2),
            'T_inc' : hp.uniform('T_inc', 4, 5),
            'T_inf' : hp.uniform('T_inf', 3, 4),
            'T_recov_severe' : hp.uniform('T_recov_severe', 5, 60),
            'P_severe' : hp.uniform('P_severe', 0.3, 0.99)
        }

        Keyword Arguments:
            model {class} -- The epi model class to be used for modelling (default: {SEIR_Testing})
            total_days {int} -- total days to simulate for (deprecated) (default: {None})
            method {str} -- Loss Method (default: {'rmse'})
            num_evals {int} -- Number of hyperopt evaluations (default: {3500})
            loss_indices {list} -- The indices of the train set to apply the losses on (default: {[-20, -10]})
            which_compartments {list} -- Which compartments to apply loss on (default: {['total_infected']})

        Returns:
            dict, hp.Trials obj -- The best params after the fit and the list of trials conducted by hyperopt
        """
        if total_days == None:
            total_days = len(df_true['date'])
        
        partial_solve_and_compute_loss = partial(self.solve_and_compute_loss, model=model,
                                                 default_params=default_params, total_days=total_days,
                                                 loss_method=method, loss_indices=loss_indices, df_true=df_true,
                                                 which_compartments=which_compartments)
        
        searchspace = variable_param_ranges
        
        trials = Trials()
        best = fmin(partial_solve_and_compute_loss,
                    space=searchspace,
                    algo=tpe.suggest,
                    max_evals=num_evals,
                    trials=trials)
        
        return best, trials
