from hyperopt import hp, tpe, fmin, Trials
from functools import partial
import importlib
import numpy as np
import copy

from main.seir.optimisers import OptimiserBase
from utils.fitting.loss import Loss_Calculator

class BO_Hyperopt(OptimiserBase):
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

        Returns:
            dict -- dict of ranges of variable params
        """

        formatted_param_ranges = {}
        for key in variable_param_ranges.keys():
            formatted_param_ranges[key] = getattr(hp, variable_param_ranges[key][1])(
                key, variable_param_ranges[key][0][0], variable_param_ranges[key][0][1])

        self.variable_param_ranges = formatted_param_ranges


    def predict(self, params_dict: dict, model, end_date=None):
        return super().predict(params_dict=params_dict, model=model, end_date=end_date)

    def predict_and_compute_loss(self, variable_params, default_params, df_true, total_days, model,
                               loss_compartments=['active', 'recovered', 'total', 'deceased'],
                               loss_weights=[1, 1, 1, 1], loss_indices=[-20, -10], loss_method='rmse',
                               debug=False):
        return super().predict_and_compute_loss(variable_params, default_params, df_true, total_days, 
                                              model, loss_compartments, loss_weights, loss_indices, 
                                              loss_method, debug)

    def init_default_params(self, df_train, default_params, train_period):
        super().init_default_params(df_train, default_params, train_period)

    def _order_trials_by_loss(self, trials_obj: Trials, sort_trials: bool = True):
        """Orders a set of trials by their corresponding loss value

        Args:
            m_dict (dict): predictions_dict

        Returns:
            array, array: Array of params and loss values resp
        """
        params_array = []
        for trial in trials_obj:
            params_dict = copy.copy(trial['misc']['vals'])
            for key in params_dict.keys():
                params_dict[key] = params_dict[key][0]
            params_array.append(params_dict)
        params_array = np.array(params_array)
        losses_array = np.array([trial['result']['loss'] for trial in trials_obj])

        if sort_trials:
            least_losses_indices = np.argsort(losses_array)
            losses_array = losses_array[least_losses_indices]
            params_array = params_array[least_losses_indices]
        return params_array, losses_array

    def forecast(self, params, train_last_date, forecast_days, model):
        return super().forecast(params, train_last_date, forecast_days, model)

    def optimise(self, num_evals=3500, train_period=28, loss_method='rmse', loss_compartments=['total'], 
                 loss_weights=[1], algo=tpe, seed=42, forecast_days=30, ** kwargs):
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
        loss_indices = [-train_period, None]
        total_days = (self.df_train.iloc[-1, :]['date'].date() -
                      self.default_params['starting_date']).days
        
        partial_predict_and_compute_loss = partial(self.predict_and_compute_loss, model=self.model,
                                                 default_params=self.default_params, total_days=total_days,
                                                 loss_method=loss_method, loss_indices=loss_indices, 
                                                 loss_weights=loss_weights, df_true=self.df_train,
                                                 loss_compartments=loss_compartments)

        algo_module = importlib.import_module(f'.{algo}', 'hyperopt')
        trials = Trials()
        rstate = np.random.RandomState(seed)
        best = fmin(partial_predict_and_compute_loss,
                    space=self.variable_param_ranges,
                    algo=algo_module.suggest,
                    max_evals=num_evals,
                    rstate=rstate,
                    trials=trials)
        
        params_array, losses_array = self._order_trials_by_loss(trials)

        partial_forecast = partial(self.forecast, 
                                   train_last_date=self.df_train.iloc[-1, :]['date'].date(),
                                   forecast_days=forecast_days,
                                   model=self.model)
        predictions_array = [partial_forecast(param) for param in params_array]

        return_dict = {
            'params': params_array,
            'predictions': predictions_array,
            'losses': losses_array
        }

        return return_dict
