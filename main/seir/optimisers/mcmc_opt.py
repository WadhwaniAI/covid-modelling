from main.seir.optimisers import OptimiserBase
from utils.fitting.loss import Loss_Calculator

from datetime import datetime
import pdb
from main.seir.uncertainty import MCMC
from functools import partial

class MCMC_Opt(OptimiserBase):
    """Class which implements all optimisation related activites (training, evaluation, etc)
    """

    def __init__(self, model, df_train, default_params, variable_param_ranges, train_period):
        self.model = model
        self.df_train = df_train
        
        self.init_default_params(df_train, default_params, 
                                 train_period=train_period)

        self.variable_param_ranges = variable_param_ranges
        self.lc = Loss_Calculator()


    def set_variable_param_ranges(self, variable_param_ranges):
        pass

    def init_default_params(self, df_train, default_params, train_period):
        super().init_default_params(df_train, default_params, train_period)
    
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

    def optimise(self, proposal_sigmas, end_date, num_evals=10000, stride=5, n_chains=10, loss_method='rmse',train_period=28 ,
                 loss_indices=[-20, -10], loss_compartments=['total'], loss_weights=[1],
                 prior='uniform', algo='gaussian', **kwargs):
        total_days = (self.df_train.iloc[-1, :]['date'].date() -
            self.default_params['starting_date']).days
        end_date = datetime.combine(
            self.df_train.iloc[-1, :]['date'].date(), datetime.min.time())
        partial_predict = partial(self.predict,model = self.model,end_date = end_date)
        partial_predict_and_compute_loss = partial_predict_and_compute_loss = partial(self.predict_and_compute_loss, model=self.model,
                                                 default_params=self.default_params, total_days=total_days,
                                                 loss_method=loss_method, loss_indices=loss_indices, 
                                                 loss_weights=loss_weights, df_true=self.df_train,
                                                 loss_compartments=loss_compartments)
        loss_indices = [-train_period, None]
        total_days = (self.df_train.iloc[-1, :]['date'].date() -
                      self.default_params['starting_date']).days
        mcmc_fit = MCMC(self.df_train, self.default_params, self.variable_param_ranges, n_chains, total_days,
                        algo, num_evals, stride, proposal_sigmas, loss_method, loss_compartments, 
                        loss_indices, loss_weights, self.model,partial_predict,partial_predict_and_compute_loss)
        mcmc_fit.run()
        
        metric = {"DIC": mcmc_fit.DIC, "GR-ratio": mcmc_fit.R_hat}
        best, tri = mcmc_fit._get_trials()
        return best, tri, metric
