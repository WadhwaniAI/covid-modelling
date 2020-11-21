import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_squared_log_error


class Loss_Calculator():

    def __init__(self):
      self.columns = ['active', 'recovered', 'deceased', 'total', 
                      'asymptomatic', 'symptomatic', 'critical',
                      'hq', 'non_o2_beds', 'o2_beds', 'icu', 'ventilator']

    def _calc_rmse(self, y_pred, y_true, log=False):
        """Calculate RMSE Loss

        Args:
            y_pred (np.array): predicted array
            y_true (np.array): true array
            log (bool, optional): If true, computes log rmse. Defaults to False.

        Returns:
            float: RMSE loss
        """
        if log:
            y_pred = np.log(y_pred[y_true > 0])
            y_true = np.log(y_true[y_true > 0])
        loss = np.sqrt(np.mean((y_true - y_pred)**2))
        return loss

    def _calc_mape(self, y_pred, y_true):
        """Calculate MAPE loss

        Args:
            y_pred (np.array): predicted array
            y_true (np.array): GT array

        Returns:
            float: MAPE loss
        """
        y_pred = y_pred[y_true != 0]
        y_true = y_true[y_true != 0]

        ape = np.abs((y_true - y_pred + 0) / y_true) *  100
        loss = np.mean(ape)
        return loss

    def _calc_smape(self, y_pred, y_true):
        y_pred = y_pred[y_true != 0]
        y_true = y_true[y_true != 0]

        ape = np.abs((y_true - y_pred + 0) / np.mean(y_true,y_pred)) *  100
        loss = np.mean(ape)
        return loss
    def _calc_l1_perc(self, y_pred, y_true,perc):
        e = y_true - y_pred
        loss = np.sum(np.max(e*perc,(perc-1)*e))
        return loss

    def _calc_mape_perc(self, y_pred, y_true,perc):
        y_pred = y_pred[y_true != 0]
        y_true = y_true[y_true != 0]
        ape = ((y_true - y_pred + 0) / y_true) *  100
        A = np.multiply(ape,perc)
        B = np.multiply(ape,perc-1)
        perc_ape = np.maximum(A,B)
        loss = np.mean(perc_ape)
        return loss

    def calc_loss_dict(self, df_prediction, df_true, method='rmse'):
        """Caclculates dict of losses for each compartment using the method specified

        Args:
            df_prediction (pd.DataFrame): prediction dataframe
            df_true (pd.DataFrame): gt dataframe
            method (str, optional): Loss method. Defaults to 'rmse'.

        Returns:
            dict: dict of loss values {compartment : loss_value}
        """
        if method == 'rmse':
            calculate = lambda x, y : self._calc_rmse(x, y)
        if method == 'rmse_log':
            calculate = lambda x, y : self._calc_rmse(x, y, log=True)
        if method == 'mape':
            calculate = lambda x, y : self._calc_mape(x, y)
        
        losses = {}
        for compartment in self.columns:
            try:
                losses[compartment] = calculate(df_prediction[compartment], df_true[compartment])
            except Exception:
                continue
        return losses

    def calc_loss(self, df_prediction, df_true, method='rmse', 
                  which_compartments=['active', 'recovered', 'total', 'deceased'], loss_weights=[1, 1, 1, 1]):
        """Calculates loss using specified method, averaged across specified compartments using specified weights

        Args:
            df_prediction (pd.DataFrame): prediction dataframe
            df_true (pd.DataFrame): gt dataframe
            method (str, optional): loss method. Defaults to 'rmse'.
            which_compartments (list, optional): Compartments to calculate loss on.
            Defaults to ['active', 'recovered', 'total', 'deceased'].
            loss_weights (list, optional): Weights for corresponding compartments. Defaults to [1, 1, 1, 1].

        Returns:
            float: loss value
        """
        losses = self.calc_loss_dict(df_prediction, df_true, method)
        loss = 0
        for i, compartment in enumerate(which_compartments):
            loss += loss_weights[i]*losses[compartment]
        return loss

    def evaluate(self, y_true, y_pred):
        """Used by IHME

        Args:
            y_true ([type]): [description]
            y_pred ([type]): [description]

        Returns:
            dict: Dict of losses
        """
        err = {}
        err['mape'] = self._calc_mape(y_true, y_pred)
        err['rmse'] = self._calc_rmse(y_true, y_pred)
        try:
            err['rmsle'] = self._calc_rmse(y_true, y_pred, log=True)
        except:
            err['rmsle'] = None
        return err

    def create_loss_dataframe_region(self, df_train, df_val, df_prediction, train_period, 
                                     which_compartments=['active', 'total']):
        """Helper function for calculating loss in training pipeline

        Arguments:
            df_train {pd.DataFrame} -- Train dataset
            df_val {pd.DataFrame} -- Val dataset
            df_prediction {pd.DataFrame} -- Model Prediction
            train_period {int} -- Length of training Period

        Keyword Arguments:
            which_compartments {list} -- List of buckets to calculate loss on (default: {['active', 'total']})

        Returns:
            pd.DataFrame -- A dataframe of train loss values and val (if val exists too)
        """
        df_loss = pd.DataFrame(columns=['train', 'val'], index=which_compartments)

        df_temp = df_prediction.loc[df_prediction['date'].isin(
            df_train['date']), ['date']+which_compartments]
        df_temp.reset_index(inplace=True, drop=True)
        df_train = df_train.loc[df_train['date'].isin(df_temp['date']), :]
        df_train.reset_index(inplace=True, drop=True)
        for compartment in df_loss.index:
            df_loss.loc[compartment, 'train'] = self._calc_mape(
                np.array(df_temp[compartment]), np.array(df_train[compartment]))

        if isinstance(df_val, pd.DataFrame):
            df_temp = df_prediction.loc[df_prediction['date'].isin(
                df_val['date']), ['date']+which_compartments]
            df_temp.reset_index(inplace=True, drop=True)
            df_val.reset_index(inplace=True, drop=True)
            for compartment in df_loss.index:
                df_loss.loc[compartment, 'val'] = self._calc_mape(
                    np.array(df_temp[compartment]), np.array(df_val[compartment]))
        else:
            del df_loss['val']
        return df_loss

    def backtesting_loss_week_by_week(self, df_prediction, df_true, method='mape', round_precision=2):
        """Implements backtesting loss (comparing unseen gt with predictions)
        Calculates the backtesting loss for each compartment, week by week into the future, 
        using the specified method

        Args:
            df_prediction (pd.DataFrame): the prediction df
            df_true (pd.DataFrame): the gt df
            method (str, optional): The loss method. Defaults to 'mape'.
            round_precision (int, optional): Precision to which we want to round the dataframe. Defaults to 2.

        Returns:
            pd.DataFrame: The loss dataframe by compartment (row), week (column)
        """
        forecast_errors_dict = {}
        week_indices = np.concatenate((np.arange(0, len(df_true), 7), [len(df_true)]))
        for i in range(len(week_indices)-1):
            df_prediction_slice = df_prediction.iloc[week_indices[i]:week_indices[i+1]-1, :]
            df_true_slice = df_true.iloc[week_indices[i]:week_indices[i+1]-1, :]

            df_prediction_slice.reset_index(drop=True, inplace=True)
            df_true_slice.reset_index(drop=True, inplace=True)
            ld = self.calc_loss_dict(df_prediction_slice, df_true_slice, method=method)

            ld = {key: round(value, round_precision) for key, value in ld.items()}
            if i+1 == len(week_indices)-1:
                days = week_indices[i+1] - week_indices[i]
                forecast_errors_dict[f'week {i+1} ({days}days)'] = ld
            else:
                forecast_errors_dict[f'week {i+1}'] = ld

        df_prediction.reset_index(drop=True, inplace=True)
        df_true.reset_index(drop=True, inplace=True)
        ld = self.calc_loss_dict(df_prediction, df_true, method=method)

        ld = {key: round(value, round_precision) for key, value in ld.items()}
        forecast_errors_dict['total'] = ld

        df = pd.DataFrame.from_dict(forecast_errors_dict)
        return df

    def create_loss_dataframe_master(self, predictions_dict, train_fit='m1'):
        starting_key = list(predictions_dict.keys())[0]
        loss_columns = pd.MultiIndex.from_product([predictions_dict[starting_key][train_fit]['df_loss'].columns, 
                                                   predictions_dict[starting_key][train_fit]['df_loss'].index])
        loss_index = predictions_dict.keys()

        df_loss_master = pd.DataFrame(columns=loss_columns, index=loss_index)
        for key in predictions_dict.keys():
            df_loss_master.loc[key, :] = np.around(
                predictions_dict[key][train_fit]['df_loss'].values.T.flatten().astype('float'), decimals=2)
            
        return df_loss_master
