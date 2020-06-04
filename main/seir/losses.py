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
                # losses[compartment] = calculate(df_prediction.iloc[-window_dict[compartment]:, compartment],
                #                                 df_true[-window_dict[compartment]:, compartment])
            except Exception:
                continue
        return losses

    def calc_loss(self, df_prediction, df_true, which_compartments=['hospitalised', 'recovered', 'total_infected', 'deceased'], method='rmse'):
        losses = self.calc_loss_dict(df_prediction, df_true, method)
        loss = 0
        for compartment in which_compartments:
            loss += losses[compartment]
        return loss
