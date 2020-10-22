
import os
import sys
import json
import datetime
import copy
import numpy as np
import pandas as pd
from functools import partial
from hyperopt import fmin, tpe, hp, Trials

sys.path.append('../../../')
from main.seir.forecast import get_forecast
from .uncertainty_base import Uncertainty
from utils.fitting.loss import Loss_Calculator
from utils.generic.enums import Columns

class MCUncertainty(Uncertainty):
    def __init__(self, predictions_dict, num_evals, variable_param_ranges, which_fit, date_of_sorting_trials, 
                 sort_trials_by_column, loss, percentiles):
        """
        Initializes uncertainty object, finds beta for distribution

        Args:
            predictions_dict (dict): predictions_dict as returned by main.seir.fitting.single_fitting_cycle
            date_of_sorting_trials (str): prediction date by which trials should be sorted + distributed
        """
        super().__init__(predictions_dict)
        self.variable_param_ranges = variable_param_ranges
        self.which_fit = which_fit
        self.date_of_sorting_trials = date_of_sorting_trials
        self.sort_trials_by_column = sort_trials_by_column
        for key in loss:
            setattr(self, key, loss[key])
        self.percentiles = percentiles
        self.beta = self.find_beta(variable_param_ranges, num_evals)
        self.beta_loss = self.avg_weighted_error({'beta': self.beta}, return_dict=True)
        self.ensemble_mean_forecast = self.avg_weighted_error({'beta': self.beta}, return_dict=False,
                                                              return_ensemble_mean_forecast=True)                                                      
        self.get_distribution()

    def trials_to_df(self, trials_processed, column=Columns.active):
        predictions = trials_processed['predictions']
        params = trials_processed['params']
        losses = trials_processed['losses']

        cols = ['loss', 'compartment']
        for key in params[0].keys():
            cols.append(key)
        trials = pd.DataFrame(columns=cols)
        for i in range(len(params)):
            to_add = copy.copy(params[i])
            to_add['loss'] = losses[i]
            to_add['compartment'] = column.name
            trials = trials.append(to_add, ignore_index=True)
        pred = pd.DataFrame(columns=predictions[0]['date'])
        for i in range(len(params)):
            pred = pred.append(predictions[i].set_index(
                'date').loc[:, [column.name]].transpose(), ignore_index=True)
        return pd.concat([trials, pred], axis=1)

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
        
        df = pd.DataFrame(columns=['loss', 'weight', 'pdf', self.date_of_sorting_trials, 'cdf'])
        df['loss'] = self.predictions_dict[self.which_fit]['trials_processed']['losses']
        df['weight'] = np.exp(-self.beta*df['loss'])
        df['pdf'] = df['weight'] / df['weight'].sum()
        df_trials = self.trials_to_df(self.predictions_dict[self.which_fit]['trials_processed'], 
                                      self.sort_trials_by_column)
        self.date_of_sorting_trials = datetime.datetime.combine(
            self.date_of_sorting_trials, datetime.time())
        df[self.date_of_sorting_trials] = df_trials.loc[:, self.date_of_sorting_trials]
        
        df = df.sort_values(by=self.date_of_sorting_trials)
        df.index.name = 'idx'
        df.reset_index(inplace=True)
        
        df['cdf'] = df['pdf'].cumsum()
        
        self.distribution = df
        return self.distribution

    def get_forecasts(self, percentiles=None):
        """
        Get forecasts at certain percentiles

        Args:
            percentiles (list, optional): percentiles at which predictions from the distribution 
                will be returned. Defaults to all deciles 10-90, as well as 2.5/97.5 and 5/95.

        Returns:
            dict: deciles_forecast, {percentile: {df_prediction: pd.DataFrame, df_loss: pd.DataFrame, params: dict}}
        """  
        if percentiles is None:
            ptile_dict = self.get_ptiles_idx(percentiles=self.percentiles)
        else:
            ptile_dict = self.get_ptiles_idx(percentiles=percentiles)
        
        deciles_forecast = {}
        
        predictions = self.predictions_dict[self.which_fit]['trials_processed']['predictions']
        params = self.predictions_dict[self.which_fit]['trials_processed']['params']
        df_district = self.predictions_dict[self.which_fit]['df_district']
        df_train_nora = df_district.set_index('date').loc[
            self.predictions_dict[self.which_fit]['df_train']['date'], :].reset_index()
        
        for key in ptile_dict.keys():
            deciles_forecast[key] = {}
            df_predictions = predictions[ptile_dict[key]]
            df_predictions['daily_cases'] = df_predictions['total'].diff()
            df_predictions.dropna(axis=0, how='any', inplace=True)
            deciles_forecast[key]['df_prediction'] = df_predictions
            deciles_forecast[key]['params'] =  params[ptile_dict[key]]
            deciles_forecast[key]['df_loss'] = Loss_Calculator().create_loss_dataframe_region(
                df_train_nora, None, df_predictions, train_period=7,
                which_compartments=self.loss_compartments)
        return deciles_forecast

    def avg_weighted_error(self, hp, return_dict=False, return_ensemble_mean_forecast=False):
        """
        Loss function to optimize beta

        Args:
            hp (dict): {'beta': float}

        Returns:
            float: average relative error calculated over trials and a val set
        """    
        beta = hp['beta']
        losses = self.predictions_dict['m1']['trials_processed']['losses']
        # This is done as rolling average on df_val has already been calculated, 
        # while df_district has no rolling average
        df_val = self.predictions_dict['m1']['df_district'].set_index('date') \
            .loc[self.predictions_dict['m1']['df_val']['date'],:]
        
        df_data_weights_val = self.predictions_dict['m1']['df_data_weights_district'].set_index('date') \
            .loc[self.predictions_dict['m1']['df_data_weights_val']['date'],:]
        
        beta_loss = np.exp(-beta*losses)

        predictions = self.predictions_dict['m1']['trials_processed']['predictions']
        allcols = self.loss_compartments
        predictions_stacked = np.array([df.loc[:, allcols].to_numpy() for df in predictions])
        predictions_stacked_weighted_by_beta = beta_loss[:, None, None] * predictions_stacked / beta_loss.sum()
        weighted_pred = np.sum(predictions_stacked_weighted_by_beta, axis=0)
        weighted_pred_df = pd.DataFrame(data=weighted_pred, columns=allcols)
        weighted_pred_df['date'] = predictions[0]['date']
        weighted_pred_df.set_index('date', inplace=True)
        weighted_pred_df_loss = weighted_pred_df.loc[weighted_pred_df.index.isin(df_val.index), :]
        
        lc = Loss_Calculator()
        if return_dict:
            return lc.calc_loss_dict(weighted_pred_df_loss, df_val, df_data_weights_val, method = self.loss_method)
        if return_ensemble_mean_forecast:
            weighted_pred_df.reset_index(inplace=True)
            return weighted_pred_df
        return lc.calc_loss(weighted_pred_df_loss, df_val, df_data_weights_val, method = self.loss_method,
                            which_compartments=allcols, loss_weights=self.loss_weights)

    def find_beta(self, variable_param_ranges, num_evals=1000):
        """
        Runs a search over m1 trials to find best beta for a probability distro

        Args:
            num_evals (int, optional): number of iterations to run hyperopt. Defaults to 1000.

        Returns:
            float: optimal beta value
        """

        for key in variable_param_ranges.keys():
            print(variable_param_ranges[key][1])
            print(variable_param_ranges[key][0][0])
            print(variable_param_ranges[key][0][1])
            
            
            variable_param_ranges[key] = getattr(hp, variable_param_ranges[key][1])(
                key, variable_param_ranges[key][0][0], variable_param_ranges[key][0][1])

        trials = Trials()
        best = fmin(self.avg_weighted_error,
                    space=variable_param_ranges,
                    algo=tpe.suggest,
                    max_evals=num_evals,
                    trials=trials)

        self.beta = best['beta']
        return self.beta

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
            raise Exception("No distribution found. Must call get_distribution first.")
        
        if percentiles is None:
            percentiles = range(10, 100, 10), np.array([2.5, 5, 95, 97.5])
            percentiles = np.sort(np.concatenate(percentiles))
        else:
            np.sort(percentiles)        

        ptile_dict = {}
        for ptile in percentiles:
            index_value = (self.distribution['cdf'] - ptile/100).apply(abs).idxmin()
            best_idx = self.distribution.loc[index_value - 2:index_value + 2, :]['idx'].min()
            ptile_dict[ptile] = int(best_idx)

        return ptile_dict
