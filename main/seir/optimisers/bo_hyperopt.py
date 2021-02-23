from hyperopt import hp, tpe, fmin, Trials

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

        return super().solve(params_dict=params_dict, model=model, end_date=end_date)


    def solve_and_compute_loss(self, variable_params, default_params, df_true, total_days, model,
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
        return super().solve_and_compute_loss(variable_params, default_params, df_true, total_days, 
                                              model, loss_compartments, loss_weights, loss_indices, 
                                              loss_method, debug)

    def optimise(*args, **kwargs):
        pass