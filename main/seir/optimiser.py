import numpy as np
import pandas as pd
import copy
from hyperopt import hp, tpe, fmin, Trials
from tqdm.notebook import tqdm

from collections import OrderedDict
import itertools
import importlib
from functools import partial, reduce
import datetime
from joblib import Parallel, delayed

from models.seir import SEIRHD
from utils.fitting.loss import Loss_Calculator

class Optimiser():
    """Class which implements all optimisation related activites (training, evaluation, etc)
    """

    def __init__(self):
        self.loss_calculator = Loss_Calculator()

    def format_variable_param_ranges(self, variable_param_ranges, fitting_method='bayes_opt'):
        """Returns the ranges for the variable params in the search space

        Keyword Arguments:

            as_str {bool} -- If true, the parameters are not returned as a hyperopt object, but as a dict in 
            string form (default: {False})

        Returns:
            dict -- dict of ranges of variable params
        """

        formatted_param_ranges = {}
        if fitting_method == 'bayes_opt':
            for key in variable_param_ranges.keys():
                formatted_param_ranges[key] = getattr(hp, variable_param_ranges[key][1])(
                    key, variable_param_ranges[key][0][0], variable_param_ranges[key][0][1])

        if fitting_method == 'gridsearch':
            for key in variable_param_ranges.keys():
                formatted_param_ranges[key] = np.linspace(variable_param_ranges[key][0][0],
                                                          variable_param_ranges[key][0][1], 
                                                          variable_param_ranges[key][1])

        return formatted_param_ranges

    def solve(self, params_dict :dict, model=SEIRHD, end_date=None):
        """This function solves the ODE for an input of params (but does not compute loss)

        Arguments:
            variable_params {dict} -- The values for the params that are variable across the searchspace
            default_params {dict} -- The values for the params that are fixed across the searchspace

        Keyword Arguments:
            model {class} -- The epi model class to be used for modelling (default: {SEIRHD})
            end_date {str} -- Last date of projection (default: {None})

        Returns:
            pd.DataFrame -- DataFrame of predictions
        """

        solver = model(**params_dict)
        total_days = (end_date.date() - params_dict['starting_date']).days
        df_prediction = solver.predict(total_days=total_days)
        return df_prediction


    # TODO add cross validation support
    def solve_and_compute_loss(self, variable_params, default_params, df_true, total_days, model=SEIRHD,
                               loss_compartments=['active', 'recovered', 'total', 'deceased'], 
                               loss_weights = [1, 1, 1, 1], loss_indices=[-20, -10], loss_method='rmse', 
                               debug=False):
        """The function that computes solves the ODE for a given set of input params and computes loss on train set

        Arguments:
            variable_params {dict} -- The values for the params that are variable across the searchspace
            default_params {dict} -- The values of the params that are fixed across the searchspace
            df_true {pd.DataFrame} -- The train set
            total_days {int} -- Total number of days into the future for which we want to simulate

        Keyword Arguments:
            model {class} -- The epi model class to be used for modelling (default: {SEIRHD})
            loss_compartments {list} -- Which compartments to apply loss on 
            (default: {['active', 'recovered', 'total', 'deceased']})
            loss_indices {list} -- Which indices of the train set to apply loss on (default: {[-20, -10]})
            loss_method {str} -- Loss Method (default: {'rmse'})
            return_dict {bool} -- If True, instead of returning single loss value, will return loss value 
            for every compartment (default: {False})

        Returns:
            float -- The loss value
        """
        params_dict = {**variable_params, **default_params}
        # Returning a very high loss value for the cases where the sampled values of probabilities are > 1
        P_keys = [k for k in params_dict.keys() if k[:2] == 'P_']
        P_values = [params_dict[k] for k in params_dict.keys() if k[:2] == 'P_']
        if sum(P_values) > 1:
            return 1e10

        # import pdb; pdb.set_trace()
        solver = model(**params_dict)
        # import pdb; pdb.set_trace()
        df_prediction = solver.predict(total_days=total_days)
        # print(df_prediction)
        # import pdb; pdb.set_trace()

        # Choose which indices to calculate loss on
        # TODO Add slicing capabilities on the basis of date
        if loss_indices == None:
            df_prediction_slice = df_prediction
            df_true_slice = df_true
        else:
            df_prediction_slice = df_prediction.iloc[loss_indices[0]:loss_indices[1], :]
            df_true_slice = df_true.iloc[loss_indices[0]:loss_indices[1], :]

        if debug:
            import pdb; pdb.set_trace()
        df_prediction_slice.reset_index(inplace=True, drop=True)
        df_true_slice.reset_index(inplace=True, drop=True)
        loss = self.loss_calculator.calc_loss(df_prediction_slice, df_true_slice, 
                                              which_compartments=loss_compartments, method=loss_method,
                                              loss_weights=loss_weights)
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

    def init_default_params(self,df_train, df_val, default_params, train_period=7):
        """Function for creating all default params for the optimisation (hyperopt/gridsearch)

        Arguments:
            df_train {pd.DataFrame} -- The train dataset

        Keyword Arguments:
            N {float} -- Population of region (default: {1e7})
            train_period {int} -- The number of days for which the model is trained (default: {7})
            observed_values {pd.Series} -- This is a row of a pandas dataframe that corresponds to the observed values 
            on the initialisation date (to initialise the latent params) (default: {None})

        Returns:
            dict -- Dict of default params
        """
        observed_values = df_train.iloc[-train_period, :]
        start_date = observed_values['date'].date()
        daily_testing = pd.DataFrame()
        
        # can move this to utils
        # import pdb; pdb.set_trace()
        try:
            daily_testing = df_train.set_index('date')[-train_period:]['tested']
            daily_testing = daily_testing.append(df_val.set_index('date')['tested'])
            tmp = copy.deepcopy(daily_testing[:-1])
            tmp.index = daily_testing[1:].index
            daily_testing[1:] -= tmp
        except Exception as e:
            pass
        #####

        extra_params = {
            'starting_date' : start_date,
            'observed_values': observed_values,
            'daily_testing' : daily_testing
        }

        return {**default_params, **extra_params}

    def gridsearch(self, df_train, default_params, variable_param_ranges, model=SEIRHD, loss_method='rmse',
                   loss_indices=[-20, -10], loss_compartments=['total'], debug=False):
        """Implements gridsearch based optimisation

        Arguments:
            df_train {pd.DataFrame} -- The train set
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
            model {class} -- The epi model class to be used for modelling (default: {SEIRHD})
            method {str} -- The loss method (default: {'rmse'})
            loss_indices {list} -- The indices on the train set to apply the loss on (default: {[-20, -10]})
            which_compartments {list} -- Which compartments to apply loss on (default: {['total']})
            debug {bool} -- If debug is true, gridsearch is not parellelised. For debugging (default: {False})

        Returns:
            arr, list(dict) -- Array of loss values, and a list of parameter dicts
        """
        total_days = (df_train.iloc[-1, :]['date'].date() - default_params['starting_date']).days

        rangelists = list(variable_param_ranges.values())
        cartesian_product_tuples = itertools.product(*rangelists)
        list_of_param_dicts = [self._create_dict(list(
            variable_param_ranges.keys()), values) for values in cartesian_product_tuples]

        partial_solve_and_compute_loss = partial(self.solve_and_compute_loss, default_params=default_params,
                                                 df_true=df_train, total_days=total_days, model=model,
                                                 loss_method=loss_method, loss_indices=loss_indices,
                                                 loss_compartments=loss_compartments, debug=False)
        
        # If debugging is enabled the gridsearch is not parallelised
        if debug:
            loss_array = []
            for params_dict in tqdm(list_of_param_dicts):
                loss_array.append(partial_solve_and_compute_loss(variable_params=params_dict))
        else:
            loss_array = Parallel(n_jobs=40)(delayed(partial_solve_and_compute_loss)(params_dict) for params_dict in tqdm(list_of_param_dicts))
                    
        return loss_array, list_of_param_dicts

    def bayes_opt(self, df_train, default_params, variable_param_ranges, model=SEIRHD, num_evals=3500, 
                  loss_method='rmse', loss_indices=[-20, -10], loss_compartments=['total'], loss_weights=[1],
                  prior='uniform', algo=tpe, **kwargs):
        """Implements Bayesian Optimisation using hyperopt library

        Arguments:
            df_train {pd.DataFrame} -- The training dataset
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
            model {class} -- The epi model class to be used for modelling (default: {SEIRHD})
            total_days {int} -- total days to simulate for (deprecated) (default: {None})
            method {str} -- Loss Method (default: {'rmse'})
            num_evals {int} -- Number of hyperopt evaluations (default: {3500})
            loss_indices {list} -- The indices of the train set to apply the losses on (default: {[-20, -10]})
            which_compartments {list} -- Which compartments to apply loss on (default: {['total']})

        Returns:
            dict, hp.Trials obj -- The best params after the fit and the list of trials conducted by hyperopt
        """
        total_days = (df_train.iloc[-1, :]['date'].date() - default_params['starting_date']).days
        
        partial_solve_and_compute_loss = partial(self.solve_and_compute_loss, model=model,
                                                 default_params=default_params, total_days=total_days,
                                                 loss_method=loss_method, loss_indices=loss_indices, 
                                                 loss_weights=loss_weights, df_true=df_train,
                                                 loss_compartments=loss_compartments)

        algo_module = importlib.import_module(f'.{algo}', 'hyperopt')
        trials = Trials()
        best = fmin(partial_solve_and_compute_loss,
                    space=variable_param_ranges,
                    algo=algo_module.suggest,
                    max_evals=num_evals,
                    trials=trials)
        
        return best, trials
