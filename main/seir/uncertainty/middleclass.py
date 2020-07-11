
import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
from functools import partial
from hyperopt import fmin, tpe, hp, Trials

sys.path.append('../../../')
from main.seir.forecast import get_forecast
from .uncertainty_base import Uncertainty
from utils.loss import Loss_Calculator
from utils.enums import Columns

class MCUncertainty(Uncertainty):
    def __init__(self, region_dict, date_of_interest):
        """
        Initializes uncertainty object, finds beta for distribution

        Args:
            region_dict (dict): region_dict as returned by main.seir.fitting.single_fitting_cycle
            date_of_interest (str): prediction date by which trials should be sorted + distributed
        """
        super().__init__(region_dict)
        self.date_of_interest = datetime.datetime.strptime(date_of_interest, '%Y-%m-%d')
        self.beta = self.find_beta(num_evals=100)
        self.get_distribution()

    def get_distribution(self):
        """
        Computes probability distribution based on given beta and date 
        over the trials in region_dict['m2']['all_trials']

        Args:

        Returns:
            pd.DataFrame: dataframe of sorted trials, with columns
                idx: original trial index
                loss: loss value for that trial
                weight: np.exp(-beta*loss)
                pdf: pdf
                cdf: cdf
                <date_of_interest>: predicted value on <date_of_interest>

        """    
        
        df = pd.DataFrame(columns=['loss', 'weight', 'pdf', self.date_of_interest, 'cdf'])
        df['loss'] = self.region_dict['m2']['trials_processed']['losses']
        df['weight'] = np.exp(-self.beta*df['loss'])
        df['pdf'] = df['weight'] / df['weight'].sum()
        df[self.date_of_interest] = self.region_dict['m2']['all_trials'].loc[:, self.date_of_interest]
        
        df = df.sort_values(by=self.date_of_interest)
        df.index.name = 'idx'
        df.reset_index(inplace=True)
        
        df['cdf'] = df['pdf'].cumsum()
        
        self.distribution = df
        return self.distribution

    def get_forecasts(self, ptile_dict=None, percentiles=None):
        """
        Get forecasts at certain percentiles

        Args:
            percentiles (list, optional): percentiles at which predictions from the distribution 
                will be returned. Defaults to all deciles 10-90, as well as 2.5/97.5 and 5/95.

        Returns:
            dict: deciles_forecast, {percentile: {df_prediction: pd.DataFrame, df_loss: pd.DataFrame, params: dict}}
        """  
        if ptile_dict is None: 
            ptile_dict = self.get_ptiles_idx(percentiles=percentiles)
        
        deciles_forecast = {}
        deciles_params = {}
        
        predictions = self.region_dict['m2']['trials_processed']['predictions']
        params = self.region_dict['m2']['trials_processed']['params']
        df_district = self.region_dict['m2']['df_district']
        df_train_nora = df_district.set_index('date').loc[self.region_dict['m2']['df_train']['date'],:].reset_index()
        
        for key in ptile_dict.keys():
            deciles_forecast[key] = {}
            df_predictions = predictions[ptile_dict[key]]
            deciles_params[key] = params[ptile_dict[key]]
            deciles_forecast[key]['df_prediction'] = df_predictions
            deciles_forecast[key]['params'] =  params[ptile_dict[key]]
            deciles_forecast[key]['df_loss'] = Loss_Calculator().create_loss_dataframe_region(
                df_train_nora, None, df_predictions, train_period=7,
                which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
        return deciles_forecast

    def avg_weighted_error(self, hp, loss_method='mape'):
        """
        Loss function to optimize beta

        Args:
            hp (dict): {'beta': float}

        Returns:
            float: average relative error calculated over trials and a val set
        """    
        beta = hp['beta']
        losses = self.region_dict['m1']['trials_processed']['losses']
        df_val = self.region_dict['m1']['df_district'].set_index('date') \
            .loc[self.region_dict['m1']['df_val']['date'],:]
        beta_loss = np.exp(-beta*losses)

        predictions = self.region_dict['m1']['trials_processed']['predictions']
        allcols = ['hospitalised', 'recovered', 'deceased', 'total_infected']
        predictions_stacked = np.array([df.loc[:, allcols].to_numpy() for df in predictions])
        predictions_stacked_weighted_by_beta = beta_loss[:, None, None] * predictions_stacked / beta_loss.sum()
        weighted_pred = np.sum(predictions_stacked_weighted_by_beta, axis=0)
        weighted_pred_df = pd.DataFrame(data=weighted_pred, columns=allcols)
        weighted_pred_df['date'] = predictions[0]['date']
        weighted_pred_df.set_index('date', inplace=True)
        weighted_pred_df = weighted_pred_df.loc[weighted_pred_df.index.isin(df_val.index), :]
        lc = Loss_Calculator()
        return lc.calc_loss(weighted_pred_df, df_val, method=loss_method)

    def find_beta(self, num_evals=1000):
        """
        Runs a search over m1 trials to find best beta for a probability distro

        Args:
            num_evals (int, optional): number of iterations to run hyperopt. Defaults to 1000.

        Returns:
            float: optimal beta value
        """    
        searchspace = {
            'beta': hp.uniform('beta', 0, 10)
        }
        trials = Trials()
        best = fmin(self.avg_weighted_error,
                    space=searchspace,
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
