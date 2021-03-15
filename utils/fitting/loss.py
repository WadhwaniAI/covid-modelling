import numpy as np
import pandas as pd


class Loss_Calculator():

    def __init__(self):
        pass

    def rmse(self, y_pred, y_true):
        """Calculate RMSE Loss

        Args:
            y_pred (np.array): predicted array
            y_true (np.array): true array

        Returns:
            float: RMSE loss
        """
        loss = np.sqrt(np.mean((y_true - y_pred)**2))
        return loss

    def rmsle(self, y_pred, y_true):
        """Calculate RMSLE Loss

        Args:
            y_pred (np.array): predicted array
            y_true (np.array): true array

        Returns:
            float: RMSLE loss
        """
        y_pred = np.log(y_pred[y_true > 0])
        y_true = np.log(y_true[y_true > 0])
        loss = np.sqrt(np.mean((y_true - y_pred)**2))
        return loss

    def mape(self, y_pred, y_true):
        """Calculate MAPE loss

        Args:
            y_pred (np.array): predicted array
            y_true (np.array): GT array

        Returns:
            float: MAPE loss
        """
        y_pred = y_pred[y_true != 0]
        y_true = y_true[y_true != 0]

        ape = np.abs((y_true - y_pred + 0) / y_true) * 100
        loss = np.mean(ape)
        return loss

    def smape(self, y_pred, y_true):
        """Function for calculating symmetric mape

        Args:
            y_pred (np.array): predicted array
            y_true (np.array): GT array

        Returns:
            float: SMAPE loss
        """
        y_pred = y_pred[y_true != 0]
        y_true = y_true[y_true != 0]

        ape = np.abs((y_true - y_pred + 0) / np.mean(y_true,y_pred)) *  100
        loss = np.mean(ape)
        return loss
    
    def qtile_l1(self, y_pred, y_true, perc):
        """Function for calculating L1 quantile loss

        Args:
            y_pred (np.array): Predicted array
            y_true (np.array): GT array
            perc (float): The quantile

        Returns:
            float: L1 Percentile Loss
        """
        e = y_true - y_pred
        loss = np.sum(np.max(e*perc,(perc-1)*e))
        return loss

    def qtile_mape(self, y_pred, y_true, perc):
        """Function for calculating MAPE quantile loss

        Args:
            y_pred (np.array): Predicted array
            y_true (np.array): GT Array
            perc (float): quantile

        Returns:
            float: MAPE Percentile Loss
        """
        y_pred = y_pred[y_true != 0]
        y_true = y_true[y_true != 0]
        ape = ((y_true - y_pred + 0) / y_true) *  100
        A = np.multiply(ape,perc)
        B = np.multiply(ape,perc-1)
        perc_ape = np.maximum(A,B)
        loss = np.mean(perc_ape)
        return loss

    def ape(self, y_pred, y_true):
        # Allow NaNs to remain
        ape = np.abs((y_true - y_pred + 0) / y_true) * 100
        return ape

    def error(self, y_pred, y_true):
        return y_pred - y_true

    def se(self, y_pred, y_true, log=False):
        if log:
            y_true = np.log(y_true)
            y_pred = np.log(y_pred)
        return self.error(y_pred, y_true)**2

    def calc_loss_dict(self, df_prediction, df_true, loss_compartments=['active', 'recovered', 'total', 'deceased'],
                       loss_method='rmse'):
        """Caclculates dict of losses for each compartment using the method specified

        Args:
            df_prediction (pd.DataFrame): prediction dataframe
            df_true (pd.DataFrame): gt dataframe
            method (str, optional): Loss method. Defaults to 'rmse'.

        Returns:
            dict: dict of loss values {compartment : loss_value}
        """
        loss_fn = getattr(self, loss_method)
        
        losses = {}
        for compartment in loss_compartments:
                losses[compartment] = loss_fn(df_prediction[compartment], df_true[compartment])
        
        return losses

    def calc_loss(self, df_prediction, df_true, loss_method='rmse', 
                  loss_compartments=['active', 'recovered', 'total', 'deceased'], 
                  loss_weights=[1, 1, 1, 1]):
        """Calculates loss using specified method, averaged across specified compartments using specified weights

        Args:
            df_prediction (pd.DataFrame): prediction dataframe
            df_true (pd.DataFrame): gt dataframe
            loss_method (str, optional): loss loss_method. Defaults to 'rmse'.
            loss_compartments (list, optional): Compartments to calculate loss on.
            Defaults to ['active', 'recovered', 'total', 'deceased'].
            loss_weights (list, optional): Weights for corresponding compartments. Defaults to [1, 1, 1, 1].

        Returns:
            float: loss value
        """
        losses = self.calc_loss_dict(df_prediction, df_true, loss_compartments, loss_method)
        loss = 0
        for i, compartment in enumerate(loss_compartments):
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
        err['mape'] = self.mape(y_true, y_pred)
        err['rmse'] = self.rmse(y_true, y_pred)
        try:
            err['rmsle'] = self.rmsle(y_true, y_pred)
        except:
            err['rmsle'] = None
        return err


    def create_loss_dataframe_region(self, df_train, df_val, df_test, df_prediction, loss_method='mape',
                                     loss_compartments=['active', 'total']):
        """Helper function for calculating loss in training pipeline

        Arguments:
            df_train {pd.DataFrame} -- Train dataset
            df_val {pd.DataFrame} -- Val dataset
            df_prediction {pd.DataFrame} -- Model Prediction

        Keyword Arguments:
            loss_compartments {list} -- List of buckets to calculate loss on (default: {['active', 'total']})

        Returns:
            pd.DataFrame -- A dataframe of train loss values and val (if val exists too)
        """
        loss_fn = getattr(self, loss_method)

        df_loss = pd.DataFrame(columns=['train', 'val', 'test'], index=loss_compartments)

        df_temp = df_prediction.loc[df_prediction['date'].isin(
            df_train['date']), ['date']+loss_compartments]
        df_temp.reset_index(inplace=True, drop=True)
        df_train = df_train.loc[df_train['date'].isin(df_temp['date']), :]
        df_train.reset_index(inplace=True, drop=True)
        for compartment in df_loss.index:
            df_loss.loc[compartment, 'train'] = loss_fn(
                np.array(df_temp[compartment]), np.array(df_train[compartment]))

        if isinstance(df_val, pd.DataFrame):
            df_temp = df_prediction.loc[df_prediction['date'].isin(
                df_val['date']), ['date']+loss_compartments]
            df_temp.reset_index(inplace=True, drop=True)
            df_val.reset_index(inplace=True, drop=True)
            for compartment in df_loss.index:
                df_loss.loc[compartment, 'val'] = loss_fn(
                    np.array(df_temp[compartment]), np.array(df_val[compartment]))
        else:
            del df_loss['val']

        if isinstance(df_test, pd.DataFrame):
            df_temp = df_prediction.loc[df_prediction['date'].isin(
                df_test['date']), ['date'] + loss_compartments]
            df_temp.reset_index(inplace=True, drop=True)
            df_test.reset_index(inplace=True, drop=True)
            for compartment in df_loss.index:
                df_loss.loc[compartment, 'test'] = self.mape(
                    np.array(df_temp[compartment]), np.array(df_test[compartment]))
        else:
            del df_loss['test']

        return df_loss

    def evaluate_pointwise(self, y_true, y_pred):
        """Used by IHME

        Args:
            y_true ([type]): [description]
            y_pred ([type]): [description]

        Returns:
            dict: Dict of losses
        """
        err = {}
        err['ape'] = self.ape(y_pred, y_true)
        err['error'] = self.error(y_pred, y_true)
        err['se'] = self.se(y_pred, y_true)
        with np.errstate(invalid='ignore'):
            err['sle'] = self.se(y_pred, y_true, log=True)
        return err

    def create_pointwise_loss_dataframe(self, y_true, y_pred):
        """Used by IHME

        Args:
            y_true ([type]): [description]
            y_pred ([type]): [description]

        Returns:
            pd.DataFrame: Dataframe of losses
        """
        loss_dict = self.evaluate_pointwise(y_true, y_pred)
        return pd.DataFrame.from_dict(loss_dict, orient='index')

    def create_pointwise_loss_dataframe_region(self, df_train, df_val, df_test, df_prediction,
                                               which_compartments=['total']):
        """

        Args:
            df_train ():
            df_val ():
            df_test ():
            df_prediction ():
            which_compartments ():

        Returns:

        """
        # TODO: Take loss as arg
        df_temp = df_prediction.loc[df_prediction['date'].isin(
            df_train['date']), ['date'] + which_compartments]
        df_temp.reset_index(inplace=True, drop=True)
        df_train = df_train.loc[df_train['date'].isin(df_temp['date']), :]
        df_train.reset_index(inplace=True, drop=True)
        loss_df_dict = {}
        for compartment in which_compartments:
            df = self.create_pointwise_loss_dataframe(
                np.array(df_train[compartment]).astype(float), np.array(df_temp[compartment]).astype(float))
            df.columns = df_train['date'].tolist()
            loss_df_dict[compartment] = df
        df_train_loss_pointwise = pd.concat(loss_df_dict.values(), axis=0, keys=which_compartments,
                                            names=['compartment', 'loss_function'])
        df_train_loss_pointwise.name = 'loss'

        df_val_loss_pointwise = None
        if isinstance(df_val, pd.DataFrame):
            df_temp = df_prediction.loc[df_prediction['date'].isin(
                df_val['date']), ['date']+which_compartments]
            df_temp.reset_index(inplace=True, drop=True)
            df_val.reset_index(inplace=True, drop=True)
            loss_df_dict = {}
            for compartment in which_compartments:
                df = self.create_pointwise_loss_dataframe(
                    np.array(df_val[compartment]).astype(float), np.array(df_temp[compartment]).astype(float))
                df.columns = df_val['date'].tolist()
                loss_df_dict[compartment] = df
            df_val_loss_pointwise = pd.concat(loss_df_dict.values(), axis=0, keys=which_compartments,
                                              names=['compartment', 'loss_function'])
            df_val_loss_pointwise.name = 'loss'

        df_test_loss_pointwise = None
        if isinstance(df_test, pd.DataFrame):
            df_temp = df_prediction.loc[df_prediction['date'].isin(
                df_test['date']), ['date'] + which_compartments]
            df_temp.reset_index(inplace=True, drop=True)
            df_test.reset_index(inplace=True, drop=True)
            loss_df_dict = {}
            for compartment in which_compartments:
                df = self.create_pointwise_loss_dataframe(
                    np.array(df_test[compartment]).astype(float), np.array(df_temp[compartment]).astype(float))
                df.columns = df_test['date'].tolist()
                loss_df_dict[compartment] = df
            df_test_loss_pointwise = pd.concat(loss_df_dict.values(), axis=0, keys=which_compartments,
                                               names=['compartment', 'loss_function'])
            df_test_loss_pointwise.name = 'loss'

        df_loss_pointwise = pd.concat([df_train_loss_pointwise, df_val_loss_pointwise, df_test_loss_pointwise],
                                      axis=0, keys=['train', 'val', 'test'],
                                      names=['split', 'compartment', 'loss_function'])

        # TODO: Check if val and/or test is None

        return df_loss_pointwise

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
            ld = self.calc_loss_dict(df_prediction_slice, df_true_slice, loss_method=method)

            ld = {key: round(value, round_precision) for key, value in ld.items()}
            if i+1 == len(week_indices)-1:
                days = week_indices[i+1] - week_indices[i]
                forecast_errors_dict[f'week {i+1} ({days}days)'] = ld
            else:
                forecast_errors_dict[f'week {i+1}'] = ld

        df_prediction.reset_index(drop=True, inplace=True)
        df_true.reset_index(drop=True, inplace=True)
        ld = self.calc_loss_dict(df_prediction, df_true, loss_method=method)

        ld = {key: round(value, round_precision) for key, value in ld.items()}
        forecast_errors_dict['total'] = ld

        df = pd.DataFrame.from_dict(forecast_errors_dict)
        return df
