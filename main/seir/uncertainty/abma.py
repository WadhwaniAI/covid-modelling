
import sys
import datetime
from copy import copy, deepcopy
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, Trials
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.append('../../../')
from main.seir.forecast import forecast_all_trials
from main.seir.optimiser import Optimiser
from .base import Uncertainty
from utils.fitting.loss import Loss_Calculator
from utils.generic.enums import Columns

class ABMAUncertainty(Uncertainty):
    def __init__(self, predictions_dict, fitting_config, forecast_config, variable_param_ranges, fitting_method, 
                 fitting_method_params, which_fit, construct_percentiles_day_wise, date_of_sorting_trials, 
                 sort_trials_by_column, loss, percentiles, process_trials=True):
        """
        Initializes uncertainty object, finds beta for distribution

        Args:
            predictions_dict (dict): predictions_dict as returned by main.seir.fitting.single_fitting_cycle
            date_of_sorting_trials (str): prediction date by which trials should be sorted + distributed
        """
        super().__init__(predictions_dict)
        # Setting all variables as class variables
        self.variable_param_ranges = variable_param_ranges
        self.which_fit = which_fit
        self.date_of_sorting_trials = date_of_sorting_trials
        self.sort_trials_by_column = sort_trials_by_column
        for key in loss:
            setattr(self, key, loss[key])
        self.percentiles = percentiles
        self.fitting_config = fitting_config
        self.forecast_config = forecast_config
        self.construct_percentiles_day_wise = construct_percentiles_day_wise
        # Processing all trials
        if process_trials:
            self.process_trials(predictions_dict, fitting_config, forecast_config)
        # Finding Best Beta
        self.beta, self.dict_of_trials = self.find_beta(
            fitting_method, fitting_method_params, variable_param_ranges)
        self.beta_loss = self.avg_weighted_error({'beta': self.beta}, return_dict=True)
        # Creating Ensemble Mean Forecast
        self.ensemble_mean_forecast = self.avg_weighted_error({'beta': self.beta}, return_dict=False,
                                                              return_ensemble_mean_forecast=True)

    def process_trials(self, predictions_dict, fitting_config, forecast_config):
        predictions_dict['m1']['trials_processed'] = forecast_all_trials(
            predictions_dict, train_fit='m1',
            model=fitting_config['model'],
            train_end_date=fitting_config['split']['end_date'],
            forecast_days=forecast_config['forecast_days']
        )

        predictions_dict['m2']['trials_processed'] = forecast_all_trials(
            predictions_dict, train_fit='m2',
            model=fitting_config['model'],
            train_end_date=fitting_config['split']['end_date'],
            forecast_days=forecast_config['forecast_days']
        )

    def trials_to_df(self, trials_processed, column=Columns.active):
        predictions = trials_processed['predictions']
        params = trials_processed['params']
        losses = trials_processed['losses']

        cols = ['loss', 'compartment']
        for key in params[0].keys():
            cols.append(key)
        trials = pd.DataFrame(columns=cols)
        for i in range(len(params)):
            to_add = copy(params[i])
            to_add['loss'] = losses[i]
            to_add['compartment'] = column.name
            trials = trials.append(to_add, ignore_index=True)
        pred = pd.DataFrame(columns=predictions[0]['date'])
        for i in range(len(params)):
            pred = pred.append(predictions[i].set_index(
                'date').loc[:, [column.name]].transpose(), ignore_index=True)
        return pd.concat([trials, pred], axis=1)

    def avg_weighted_error(self, params, return_dict=False, return_ensemble_mean_forecast=False):
        """
        Loss function to optimize beta

        Args:
            params (dict): {'beta': float}

        Returns:
            float: average relative error calculated over trials and a val set
        """    
        beta = params['beta']
        losses = self.predictions_dict['m1']['trials_processed']['losses']
        # This is done as rolling average on df_val has already been calculated, 
        # while df_district has no rolling average
        df_val = self.predictions_dict['m1']['df_district'].set_index('date') \
            .loc[self.predictions_dict['m1']['df_val']['date'],:]
        beta_loss = np.exp(-beta*losses)

        predictions = self.predictions_dict['m1']['trials_processed']['predictions']
        loss_cols = self.loss_compartments
        allcols = ['total', 'active', 'recovered', 'deceased', 'hq', 'non_o2_beds', 'o2_beds', 'icu', 'ventilator',
                   'asymptomatic', 'symptomatic', 'critical']
        allcols = list(set(allcols) & set(predictions[0].columns))
        shapes = np.array([list(df.loc[:, allcols].to_numpy().shape) for df in predictions])
        correct_shape_idxs = np.where(shapes[:, 0] == np.amax(shapes, axis=0)[0])[0]
        pruned_predictions = [df for i, df in enumerate(predictions) if i in correct_shape_idxs]
        pruned_losses = beta_loss[correct_shape_idxs]
        predictions_stacked = np.stack([df.loc[:, allcols].to_numpy() for df in pruned_predictions], axis=0)
        predictions_stacked_weighted_by_beta = pruned_losses[:, None, None] * predictions_stacked / pruned_losses.sum()
        weighted_pred = np.sum(predictions_stacked_weighted_by_beta, axis=0)
        weighted_pred_df = pd.DataFrame(data=weighted_pred, columns=allcols)
        weighted_pred_df['date'] = predictions[0]['date']
        weighted_pred_df.set_index('date', inplace=True)
        weighted_pred_df_loss = weighted_pred_df.loc[weighted_pred_df.index.isin(df_val.index), :]
        lc = Loss_Calculator()
        if return_dict:
            return lc.calc_loss_dict(weighted_pred_df_loss, df_val, method=self.loss_method)
        if return_ensemble_mean_forecast:
            weighted_pred_df.reset_index(inplace=True)
            return weighted_pred_df
        return lc.calc_loss(weighted_pred_df_loss, df_val, method=self.loss_method,
                            which_compartments=loss_cols, loss_weights=self.loss_weights)

    def find_beta(self, fitting_method, fitting_method_params, variable_param_ranges):
        """
        Runs a search over m1 trials to find best beta for a probability distro

        Args:
            num_evals (int, optional): number of iterations to run hyperopt. Defaults to 1000.

        Returns:
            float: optimal beta value
        """
        op = Optimiser()
        formatted_searchspace = op.format_variable_param_ranges(
            variable_param_ranges, fitting_method)

        if fitting_method == 'bayes_opt':
            trials = Trials()
            best = fmin(self.avg_weighted_error,
                        space=formatted_searchspace,
                        algo=tpe.suggest,
                        max_evals=fitting_method_params['num_evals'],
                        trials=trials)

            return best['beta'], trials
        elif fitting_method == 'gridsearch':
            if fitting_method_params['parallelise']:
                loss_values = Parallel(n_jobs=40)(delayed(self.avg_weighted_error)(
                    {'beta': beta_value}) for beta_value in tqdm(formatted_searchspace['beta']))
            else:
                loss_values = [self.avg_weighted_error({'beta': beta_value})
                               for beta_value in tqdm(formatted_searchspace['beta'])]
            min_loss, best_beta = (np.min(loss_values), 
                                   formatted_searchspace['beta'][np.argmin(loss_values)])
            dict_of_trials = dict(zip(formatted_searchspace['beta'], loss_values))
            print(f'Best beta - {best_beta}')
            print(f'Min Loss - {min_loss}')
            return best_beta, dict_of_trials


    def get_distribution(self):
        """
        Computes probability distribution based on given beta and date 
        over the trials in predictions_dict[self.which_fit]['all_trials']

        Args:

        Returns:
            pd.DataFrame: dataframe of sorted trials, with columns
                idx: original trial index
                loss: loss value for that trial
                weight: np.exp(-beta*loss)
                pdf: pdf
                cdf: cdf
                <date_of_sorting_trials>: predicted value on <date_of_sorting_trials>

        """    
        
        df = pd.DataFrame(columns=['loss', 'weight', 'pdf'])
        df['loss'] = self.predictions_dict[self.which_fit]['trials_processed']['losses']
        df['weight'] = np.exp(-self.beta*df['loss'])
        df['pdf'] = df['weight'] / df['weight'].sum()
        df_trials = self.trials_to_df(self.predictions_dict[self.which_fit]['trials_processed'], 
                                      self.sort_trials_by_column)
        # Removing non time stamp columns
        df_trials = df_trials.loc[:, [x for x in df_trials.columns if type(x) is pd.Timestamp]]
        # Converting to datetime
        df_trials.columns = [datetime.datetime.date(x) if type(x) is pd.Timestamp else x for x in df_trials.columns]
        df = pd.concat([df, df_trials], axis=1)
        df.index.name = 'idx'
        df.reset_index(inplace=True)
        return df

    def get_forecasts(self, percentiles=None):
        """
        Get forecasts at certain percentiles

        Args:
            percentiles (list, optional): percentiles at which predictions from the distribution 
                will be returned. Defaults to all deciles 10-90, as well as 2.5/97.5 and 5/95.

        Returns:
            dict: deciles_forecast, {percentile: {df_prediction: pd.DataFrame, df_loss: pd.DataFrame, params: dict}}
        """ 
        # Getting Distribution (decile distributions)
        self.distribution = self.get_distribution()

        if percentiles is None:
            df_ptile_idxs = self.get_ptiles_idx(percentiles=self.percentiles)
        else:
            df_ptile_idxs = self.get_ptiles_idx(percentiles=percentiles)
                
        deciles_forecast = {}
        
        predictions = self.predictions_dict[self.which_fit]['trials_processed']['predictions']
        predictions = [df.loc[:, :'total'] for df in predictions]
        predictions_stacked = np.stack([df.to_numpy() for df in predictions], axis=0)
        params = self.predictions_dict[self.which_fit]['trials_processed']['params']
        df_district = self.predictions_dict[self.which_fit]['df_district']
        df_train_nora = df_district.set_index('date').loc[
            self.predictions_dict[self.which_fit]['df_train']['date'], :].reset_index()
        
        for ptile in df_ptile_idxs.columns:
            deciles_forecast[ptile] = {}
            if self.construct_percentiles_day_wise:
                ptile_pred_stacked = predictions_stacked[df_ptile_idxs[ptile].to_list(), :, :]
                pred_data = ptile_pred_stacked.diagonal().T
                df_prediction = pd.DataFrame(data=pred_data, columns=predictions[0].columns)
            else:
                df_prediction = predictions[df_ptile_idxs.iloc[0][ptile]]
            
            df_prediction.loc[:, 'S':] = df_prediction.loc[:, 'S':].apply(pd.to_numeric)
            deciles_forecast[ptile]['df_prediction'] = df_prediction
            if not self.construct_percentiles_day_wise:
                deciles_forecast[ptile]['params'] = params[df_ptile_idxs.iloc[0][ptile]]
            deciles_forecast[ptile]['df_loss'] = Loss_Calculator().create_loss_dataframe_region(
                df_train_nora, None, df_prediction, train_period=self.fitting_config['split']['val_period'],
                which_compartments=self.loss_compartments)
        return deciles_forecast

    def _ptile_idx_helper(self, percentiles, date):
        df = deepcopy(self.distribution)
        df = df.sort_values(by=date)
        df.reset_index(drop=True, inplace=True)

        df['cdf'] = df['pdf'].cumsum()

        ptile_idxs = []
        for ptile in percentiles:
            index_value = (df['cdf'] - ptile/100).apply(abs).idxmin()
            ptile_idxs.append(df.loc[index_value, 'idx'])

        return ptile_idxs

    def get_ptiles_idx(self, percentiles=None):
        """
        Get the predictions at certain percentiles from a distribution of trials

        Args:
            percentiles (list, optional): percentiles at which predictions from the distribution 
                will be returned. Defaults to all deciles 10-90, as well as 2.5/97.5 and 5/95.

        Returns:
            dict: {percentile: index} where index is the trial index (to arrays in predictions_dict)
        """
        if self.distribution is None:
            raise Exception(
                "No distribution found. Must call get_distribution first.")

        if percentiles is None:
            percentiles = np.concatenate(
                range(10, 100, 10), np.array([2.5, 5, 95, 97.5]))
        percentiles = np.sort(percentiles)

        dates = [x for x in self.distribution.columns if isinstance(x, datetime.date)]
        df_ptile_idxs = pd.DataFrame(columns=percentiles, index=dates)

        if self.construct_percentiles_day_wise:
            for date, _ in df_ptile_idxs.iterrows():
                df_ptile_idxs.loc[date, :] = self._ptile_idx_helper(percentiles, date)
        else:
            df_ptile_idxs.loc[self.date_of_sorting_trials, :] = self._ptile_idx_helper(
                percentiles, self.date_of_sorting_trials)
            df_ptile_idxs = df_ptile_idxs.loc[[self.date_of_sorting_trials], :]
        
        return df_ptile_idxs
