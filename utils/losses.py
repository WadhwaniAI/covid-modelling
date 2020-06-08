import numpy as np
import pandas as pd

class Loss_Calculator():

    def _calc_rmse(self, y_pred, y_true, log=False):
        if log:
            y_true = np.log(y_true)
            y_pred = np.log(y_pred)
        loss = np.sqrt(np.mean((y_true - y_pred)**2))
        return loss

    def _calc_mape(self, y_pred, y_true):
        y_pred = y_pred[y_true > 0]
        y_true = y_true[y_true > 0]

        ape = np.abs((y_true - y_pred + 0) / y_true) *  100
        loss = np.mean(ape)
        return loss

    def calc_loss_dict(self, df_prediction, df_true, method='rmse'):
        if method == 'rmse':
            calculate = lambda x, y : self._calc_rmse(x, y)
        if method == 'rmse_log':
            calculate = lambda x, y : self._calc_rmse(x, y, log=True)
        if method == 'mape':
            calculate = lambda x, y : self._calc_mape(x, y)
        
        losses = {}
        for compartment in ['hospitalised', 'recovered', 'deceased', 'total_infected']:
            try:
                losses[compartment] = calculate(df_prediction[compartment], df_true[compartment])
            except Exception:
                continue
        return losses

    def calc_loss(self, df_prediction, df_true, which_compartments=['hospitalised', 'recovered', 'total_infected', 'deceased'], method='rmse'):
        losses = self.calc_loss_dict(df_prediction, df_true, method)
        loss = 0
        for compartment in which_compartments:
            loss += losses[compartment]
        return loss

    def evaluate(y_true, y_pred):
        err = {}
        err['mape'] = mape(y_true, y_pred)
        err['rmse'] = rmse(y_true, y_pred)
        try:
            err['rmsle'] = rmsle(y_true, y_pred)
        except:
            err['rmsle'] = None
        return err

    def create_loss_dataframe_region(self, df_train, df_val, df_prediction, train_period, 
                       which_compartments=['hospitalised', 'total_infected']):
        """Helper function for calculating loss in training pipeline

        Arguments:
            df_train {pd.DataFrame} -- Train dataset
            df_val {pd.DataFrame} -- Val dataset
            df_prediction {pd.DataFrame} -- Model Prediction
            train_period {int} -- Length of training Period

        Keyword Arguments:
            which_compartments {list} -- List of buckets to calculate loss on (default: {['hospitalised', 'total_infected']})

        Returns:
            pd.DataFrame -- A dataframe of train loss values and val (if val exists too)
        """
        loss_calculator = Loss_Calculator()
        df_loss = pd.DataFrame(columns=['train', 'val'], index=which_compartments)

        df_temp = df_prediction.loc[df_prediction['date'].isin(df_train['date']), [
            'date', 'hospitalised', 'total_infected', 'deceased', 'recovered']]
        df_temp.reset_index(inplace=True, drop=True)
        df_train.reset_index(inplace=True, drop=True)
        for compartment in df_loss.index:
            df_loss.loc[compartment, 'train'] = loss_calculator._calc_mape(
                np.array(df_train[compartment].iloc[-train_period:]), np.array(df_temp[compartment].iloc[-train_period:]))

        if isinstance(df_val, pd.DataFrame):
            df_temp = df_prediction.loc[df_prediction['date'].isin(df_val['date']), [
                'date', 'hospitalised', 'total_infected', 'deceased', 'recovered']]
            df_temp.reset_index(inplace=True, drop=True)
            df_val.reset_index(inplace=True, drop=True)
            for compartment in df_loss.index:
                df_loss.loc[compartment, 'val'] = loss_calculator._calc_mape(
                    np.array(df_val[compartment]), np.array(df_temp[compartment]))
        else:
            del df_loss['val']
        return df_loss
