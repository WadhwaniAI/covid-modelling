
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
        self.beta = self.find_beta(num_evals=1000)
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
        df['loss'] = self.region_dict['m2']['losses']
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
        
        predictions = self.region_dict['m2']['predictions']
        params = self.region_dict['m2']['params']
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

    def avg_weighted_error(self, hp):
        """
        Loss function to optimize beta

        Args:
            hp (dict): {'beta': float}

        Returns:
            float: average relative error calculated over trials and a val set
        """    
        beta = hp['beta']
        losses = self.region_dict['m1']['losses']
        df_val = self.region_dict['m1']['df_district'].set_index('date') \
            .loc[self.region_dict['m1']['df_val']['date'],:]
        active_predictions = self.region_dict['m1']['all_trials'].loc[:, df_val.index]
        beta_loss = np.exp(-beta*losses)
        avg_rel_err = 0
        for date in df_val.index:
            weighted_pred = (beta_loss*active_predictions[date]).sum() / beta_loss.sum()
            rel_error = (weighted_pred - df_val.loc[date,'hospitalised']) / df_val.loc[date,'hospitalised']
            avg_rel_err += abs(rel_error)
        avg_rel_err /= len(df_val)
        return avg_rel_err

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
