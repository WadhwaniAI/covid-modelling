import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_squared_log_error


class Loss_Calculator():

    def __init__(self):
      self.columns = ['active', 'recovered', 'deceased', 'total', 
                      'asymptomatic', 'symptomatic', 'critical', 'ccc2', 'dchc', 'dch',
                      'hq', 'non_o2_beds', 'o2_beds', 'icu', 'ventilator']

    def _calc_rmse(self, y_pred, y_true, log=False):
        if log:
            y_true = np.log(y_true[y_true > 0])
            y_pred = np.log(y_pred[y_true > 0])
        loss = np.sqrt(np.mean((y_true - y_pred)**2))
        return loss

    def _calc_mape(self, y_pred, y_true):
        y_pred = y_pred[y_true != 0]
        y_true = y_true[y_true != 0]

        ape = np.abs((y_true - y_pred + 0) / y_true) *  100
        loss = np.mean(ape)
        return loss

    def _calc_wape(self, y_pred, y_true, temporal_weights):

        y_pred = np.array(y_pred)[y_true != 0]
        y_true = np.array(y_true)[y_true != 0]

        temporal_weights = np.array(temporal_weights)

        # my code
        # print('my code to check calc wape')
        # print(temporal_weights)
        # print(y_true)
        # print(y_pred)

        ape = np.abs((y_true - y_pred + 0) * temporal_weights / y_true * temporal_weights) 
        loss = np.mean(ape)
        return loss

    def calc_loss_dict(self, df_prediction, df_true, method='rmse', temporal_weights = None):
        if method == 'rmse':
            calculate = lambda x, y : self._calc_rmse(x, y)
        if method == 'rmse_log':
            calculate = lambda x, y : self._calc_rmse(x, y, log=True)
        if method == 'mape':
            calculate = lambda x, y : self._calc_mape(x, y)
        if method == 'wape' :
            # my code
            # print("I AM HERE IN THE CALC LOSS DICT FUNCTION")
            calculate = lambda x, y : self._calc_wape(x, y, temporal_weights)

        losses = {}
        for compartment in self.columns:
            # my code
            # losses[compartment] = calculate(df_prediction[compartment], df_true[compartment])
            
            try:
                losses[compartment] = calculate(df_prediction[compartment], df_true[compartment])
            except Exception:
                continue
                
        return losses

    def calc_loss(self, df_prediction, df_true, method='rmse', 
                  which_compartments=['active', 'recovered', 'total', 'deceased'], 
                  loss_weights=[1, 1, 1, 1], loss_temporal_weights=None):
        
        # my code
        # print('df_true', df_true)
        # print('df_pred', df_prediction)
        
        losses = self.calc_loss_dict(df_prediction, df_true, method, temporal_weights=loss_temporal_weights)
        loss = 0
        # my code
        # print("LOSSES", losses)
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
