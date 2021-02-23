from abc import ABC, abstractmethod

from utils.fitting.loss import Loss_Calculator
from datetime import datetime, timedelta

class OptimiserBase(ABC):
    """Class which implements all optimisation related activites (training, evaluation, etc)
    """

    def __init__(self):
        self.lc = Loss_Calculator()

    @abstractmethod
    def set_variable_param_ranges(self, variable_param_ranges):
        pass

    @abstractmethod
    def init_default_params(self, df_train, default_params, train_period):
        observed_values = df_train.iloc[-train_period, :]
        start_date = observed_values['date'].date()

        extra_params = {
            'starting_date': start_date,
            'observed_values': observed_values
        }

        self.default_params = {**default_params, **extra_params}


    @abstractmethod
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

        solver = model(**params_dict)
        total_days = (end_date.date() - params_dict['starting_date']).days
        if total_days < 0:
            raise Exception('Number of days of prediction cannot be negative.')
        df_prediction = solver.predict(total_days=total_days)
        return df_prediction


    # TODO add cross validation support
    @abstractmethod
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
        params_dict = {**variable_params, **default_params}

        # Returning a very high loss value for the cases where the sampled values of probabilities are > 1
        P_values = [params_dict[k]
                    for k in params_dict.keys() if k[:2] == 'P_']
        if sum(P_values) > 1:
            return 1e10

        solver = model(**params_dict)
        df_prediction = solver.predict(total_days=total_days)

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
        loss = self.lc.calc_loss(df_prediction_slice, df_true_slice,
                                 loss_compartments=loss_compartments,
                                 loss_method=loss_method,
                                 loss_weights=loss_weights)
        return loss

    @abstractmethod
    def forecast(self, params, train_last_date, forecast_days, model):
        simulate_till = train_last_date + timedelta(days=forecast_days)
        simulate_till = datetime.combine(simulate_till, datetime.min.time())

        df_prediction = self.predict({**params, **self.default_params}, model=model,
                                     end_date=simulate_till)
        return df_prediction

    @abstractmethod
    def optimise(*args, **kwargs):
        pass
