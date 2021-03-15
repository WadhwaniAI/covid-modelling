import numpy as np

from tqdm import tqdm

import itertools
from functools import partial
from datetime import timedelta
from joblib import Parallel, delayed

from main.seir.optimisers import OptimiserBase
from utils.fitting.loss import Loss_Calculator

class GridSearch(OptimiserBase):
    """Class which implements all optimisation related activites (training, evaluation, etc)
    """

    def __init__(self, model, df_train, default_params, variable_param_ranges, train_period):
        self.model = model
        self.df_train = df_train
        self.init_default_params(df_train, default_params,
                                 train_period=train_period)

        self.set_variable_param_ranges(variable_param_ranges)
        self.lc = Loss_Calculator()

    def set_variable_param_ranges(self, variable_param_ranges):
        """Returns the ranges for the variable params in the search space

        Keyword Arguments:

            as_str {bool} -- If true, the parameters are not returned as a hyperopt object, but as a dict in
            string form (default: {False})

        Returns:
            dict -- dict of ranges of variable params
        """

        formatted_param_ranges = {}
        for key in variable_param_ranges.keys():
            formatted_param_ranges[key] = np.linspace(variable_param_ranges[key][0][0],
                                                      variable_param_ranges[key][0][1],
                                                      variable_param_ranges[key][1])

        self.variable_param_ranges = formatted_param_ranges

    def init_default_params(self, df_train, default_params, train_period):
        return super().init_default_params(df_train, default_params, train_period)

    def predict(self, params_dict: dict, model, end_date=None):
        return super().predict(params_dict=params_dict, model=model, end_date=end_date)

    def predict_and_compute_loss(self, variable_params, default_params, df_true, total_days, model,
                               loss_compartments=['active', 'recovered', 'total', 'deceased'],
                               loss_weights=[1, 1, 1, 1], loss_indices=[-20, -10], loss_method='rmse',
                               debug=False):
        return super().predict_and_compute_loss(variable_params, default_params, df_true, total_days,
                                              model, loss_compartments, loss_weights, loss_indices,
                                              loss_method, debug)

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

    def forecast(self, params, train_last_date, forecast_days, model):
        return super().forecast(params, train_last_date, forecast_days, model)

    def optimise(self, loss_method='rmse', loss_indices=[-20, -10], loss_compartments=['total'], 
                 train_period=28, val_period=3, forecast_days=30, parallelise=False, n_threads=40, **kwargs):
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
            loss_method {str} -- The loss method (default: {'rmse'})
            loss_indices {list} -- The indices on the train set to apply the loss on (default: {[-20, -10]})
            loss_compartments {list} -- Which compartments to apply loss on (default: {['total']})
            debug {bool} -- If debug is true, gridsearch is not parellelised. For debugging (default: {False})

        Returns:
            arr, list(dict) -- Array of loss values, and a list of parameter dicts
        """
        total_days = (self.df_train.iloc[-1, :]['date'].date() -
                      self.default_params['starting_date']).days

        rangelists = list(self.variable_param_ranges.values())
        cartesian_product_tuples = itertools.product(*rangelists)
        params_array = [self._create_dict(list(
            self.variable_param_ranges.keys()), values) for values in cartesian_product_tuples]

        partial_predict_and_compute_loss = partial(self.predict_and_compute_loss, 
                                                 default_params=self.default_params,
                                                 df_true=self.df_train, total_days=total_days, 
                                                 model=self.model, loss_method=loss_method, 
                                                 loss_indices=loss_indices,
                                                 loss_compartments=loss_compartments, debug=False)
        
        # If debugging is enabled the gridsearch is not parallelised
        if not parallelise:
            losses_array = []
            for params_dict in tqdm(params_array):
                losses_array.append(partial_predict_and_compute_loss(variable_params=params_dict))
        else:
            losses_array = Parallel(n_jobs=n_threads)(delayed(partial_predict_and_compute_loss)(
                params_dict) for params_dict in tqdm(params_array))


        train_last_date = self.df_train.iloc[-1, :]['date'].date() + \
            timedelta(days=val_period)
        partial_forecast = partial(self.forecast,
                                   train_last_date=train_last_date,
                                   forecast_days=forecast_days,
                                   model=self.model)

        predictions_array = [partial_forecast(param) for param in params_array]

        return_dict = {
            'params': params_array,
            'predictions': predictions_array,
            'losses': np.array(losses_array)
        }

        return return_dict
