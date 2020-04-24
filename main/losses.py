import numpy as np
import pandas as pd

class Loss_Calculator():

    def _calc_rmse(y_pred, y_true, log=True):
        if log:
            y_true = np.log(y_true)
            y_pred = np.log(y_pred)
        loss = np.sqrt(np.mean((y_true - y_pred)**2))
        return loss

    def _calc_mape(y_pred, y_true):
        y_pred = y_pred[y_true > 0]
        y_true = y_true[y_true > 0]

        ape = np.abs((y_true - y_pred + 0) / y_true) *  100
        loss = np.mean(ape)
        return loss

    def calc_loss_dict(states_time_matrix, df, method='rmse', rmse_log=False):
        pred_hospitalisations = states_time_matrix[6] + states_time_matrix[7] + states_time_matrix[8]
        pred_recoveries = states_time_matrix[9]
        pred_fatalities = states_time_matrix[10]
        pred_infectious_unknown = states_time_matrix[2] + states_time_matrix[4]
        pred_total_cases = pred_hospitalisations + pred_recoveries + pred_fatalities
        
        if method == 'rmse':
            if rmse_log:
                calculate = lambda x, y : self._calc_rmse(x, y)
            else:
                calculate = lambda x, y : self._calc_rmse(x, y, log=False)
        
        if method == 'mape':
                calculate = lambda x, y : self._calc_mape(x, y)
        
        losses = {}
        losses['hospitalised'] = calculate(pred_hospitalisations, df['Hospitalised'])
        losses['recovered'] = calculate(pred_recoveries, df['Recovered'])
        losses['fatalities'] = calculate(pred_fatalities, df['Fatalities'])
        losses['active_infections'] = calculate(pred_infectious_unknown, df['Unknown Active Infections'])
        losses['total'] = calculate(pred_total_cases, df['Total Infected'])
        
        return losses


    def calc_loss(states_time_matrix, df, which_compartments=['hospitalised', 'recovered', 'total', 'fatalities'], method='rmse', rmse_log=False):
        losses = self.calc_loss_dict(states_time_matrix, df, method, rmse_log)
        loss = 0
        for compartment in which_compartments:
            loss += losses[compartment]
        return loss