from main.seir.optimisers import OptimiserBase
from utils.fitting.loss import Loss_Calculator

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from functools import partial
import numpy as np

class BO_SKOpt(OptimiserBase):
    """Class which implements all optimisation related activites (training, evaluation, etc)
    """

    def __init__(self, model, df_train, default_params, variable_param_ranges, train_period):
        self.model = model
        self.df_train = df_train
        self.init_default_params(df_train, default_params,
                                 train_period=train_period)

        self.set_variable_param_ranges(variable_param_ranges)
        self.lc = Loss_Calculator()

    def init_default_params(self, df_train, default_params, train_period):
        super().init_default_params(df_train, default_params, train_period)

    def set_variable_param_ranges(self, variable_param_ranges):
        formatted_param_ranges = []
        for key in variable_param_ranges.keys():
            formatted_param_ranges.append(
                Real(
                    name=key, 
                    low=variable_param_ranges[key][0][0], 
                    high=variable_param_ranges[key][0][1],
                    prior=variable_param_ranges[key][1])
            )

        self.variable_param_ranges = formatted_param_ranges


    def predict(self, params_dict: dict, model, end_date=None):
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

        return super().predict(params_dict=params_dict, model=model, end_date=end_date)


    def predict_and_compute_loss(self, variable_params, default_params, df_true, total_days, model,
                                 loss_compartments=['active', 'recovered', 'total', 'deceased'],
                                 loss_weights=[1, 1, 1, 1], loss_indices=[-20, -10], loss_method='rmse',
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
        return super().predict_and_compute_loss(variable_params, default_params, df_true, total_days, 
                                              model, loss_compartments, loss_weights, loss_indices, 
                                              loss_method, debug)

    def forecast(self, params, train_last_date, forecast_days, model):
        return super().forecast(params, train_last_date, forecast_days, model)

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

    def _order_trials_by_loss(self, losses_array, params_array, predictions_array):
        least_losses_indices = np.argsort(losses_array)
        losses_array = losses_array[least_losses_indices]
        params_array = np.array(params_array)
        params_array = params_array[least_losses_indices]
        predictions_array = [predictions_array[i] for i in least_losses_indices]

        return losses_array, params_array, predictions_array

    def optimise(self, train_period=28, loss_method='rmse', loss_compartments=['total'],
                 loss_weights=[1], acq_func="EI", n_calls=15, n_initial_points=5, noise=0.1**2,     
                 seed=1234, forecast_days=30, ** kwargs):

        loss_indices = [-train_period, None]
        total_days = (self.df_train.iloc[-1, :]['date'].date() -
                      self.default_params['starting_date']).days

        @use_named_args(dimensions=self.variable_param_ranges)
        def predict_and_compute_loss_skopt(**kwargs):
            variable_params = kwargs
            return self.predict_and_compute_loss(variable_params, self.default_params, self.df_train, 
                                                total_days, self.model, loss_compartments, loss_weights, 
                                                loss_indices, loss_method, debug=False)

        res = gp_minimize(predict_and_compute_loss_skopt,
                          dimensions=self.variable_param_ranges,
                          acq_func=acq_func,    
                          n_calls=n_calls,       
                          n_initial_points=n_initial_points,
                          noise=noise,     
                          random_state=seed)

        losses_array = res['func_vals']
        params_matrix = np.array(res['x_iters'])
        param_names = [x.name for x in self.variable_param_ranges]
        params_array = [self._create_dict(param_names, value) for value in params_matrix]
        
        partial_forecast = partial(self.forecast,
                                   train_last_date=self.df_train.iloc[-1,:]['date'].date(),
                                   forecast_days=forecast_days,
                                   model=self.model)

        predictions_array = [partial_forecast(param) for param in params_array]

        losses_array, params_array, predictions_array = self._order_trials_by_loss(
            losses_array, params_array, predictions_array)

        return_dict = {
            'params': params_array,
            'predictions': predictions_array,
            'losses': losses_array
        }

        return return_dict
