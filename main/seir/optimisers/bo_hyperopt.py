from hyperopt import hp, tpe, fmin, Trials
from functools import partial
import importlib
import numpy as np

from main.seir.optimisers import OptimiserBase
from utils.fitting.loss import Loss_Calculator

class BO_Hyperopt(OptimiserBase):
    """Class which implements all optimisation related activites (training, evaluation, etc)
    """

    def __init__(self):
        self.lc = Loss_Calculator()

    def format_variable_param_ranges(self, variable_param_ranges):
        """Returns the ranges for the variable params in the search space

        Returns:
            dict -- dict of ranges of variable params
        """

        formatted_param_ranges = {}
        for key in variable_param_ranges.keys():
            formatted_param_ranges[key] = getattr(hp, variable_param_ranges[key][1])(
                key, variable_param_ranges[key][0][0], variable_param_ranges[key][0][1])

        return formatted_param_ranges


    def solve(self, params_dict: dict, model, end_date=None):
        return super().solve(params_dict=params_dict, model=model, end_date=end_date)

    def solve_and_compute_loss(self, variable_params, default_params, df_true, total_days, model,
                               loss_compartments=['active', 'recovered', 'total', 'deceased'],
                               loss_weights=[1, 1, 1, 1], loss_indices=[-20, -10], loss_method='rmse',
                               debug=False):
        return super().solve_and_compute_loss(variable_params, default_params, df_true, total_days, 
                                              model, loss_compartments, loss_weights, loss_indices, 
                                              loss_method, debug)

    def init_default_params(self, df_train, default_params, train_period):
        return super().init_default_params(df_train, default_params, train_period)

    def optimise(self, df_train, default_params, variable_param_ranges, model, num_evals=3500, 
                 loss_method='rmse', loss_indices=[-20, -10], loss_compartments=['total'], 
                 loss_weights=[1], algo=tpe, seed=42, **kwargs):
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
        rstate = np.random.RandomState(seed)
        best = fmin(partial_solve_and_compute_loss,
                    space=variable_param_ranges,
                    algo=algo_module.suggest,
                    max_evals=num_evals,
                    rstate=rstate,
                    trials=trials)
        
        return best, trials
