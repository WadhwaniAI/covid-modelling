from datetime import datetime

import pandas as pd

from main.seir.uncertainty import Uncertainty
from utils.loss import Loss_Calculator


class MCMCUncertainty(Uncertainty):

    def __init__(self, region_dict, date_of_interest):
        super().__init__(region_dict)
        self.trials = None
        self.date_of_interest = datetime.strptime(date_of_interest, '%Y-%m-%d')
        self.get_distribution()

    def get_distribution(self):
        df = pd.DataFrame(columns=[self.date_of_interest, 'cdf'])
        df[self.date_of_interest] = self.region_dict['m2']['all_trials'].loc[:, self.date_of_interest]

        df = df.sort_values(by=self.date_of_interest)
        df.index.name = 'idx'
        df.reset_index(inplace=True)

        df['cdf'] = [(x*100)/len(df['idx']) for x in range(1, len(df['idx'])+1)]

        self.distribution = df
        return self.distribution

    def get_forecasts(self, ptile_dict=None, percentiles=None, train_period=7):
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

        predictions = self.region_dict['m2']['trials_processed']['predictions']
        params = self.region_dict['m2']['trials_processed']['params']
        df_district = self.region_dict['m2']['df_district']
        df_train_nora = df_district.set_index('date').loc[self.region_dict['m2']['df_train']['date'], :].reset_index()

        for key in ptile_dict.keys():
            deciles_forecast[key] = {}
            df_predictions = predictions[ptile_dict[key]]
            df_predictions['daily_cases'] = df_predictions['total_infected'].diff()
            df_predictions.dropna(axis=0, how='any', inplace=True)
            deciles_forecast[key]['df_prediction'] = df_predictions
            deciles_forecast[key]['params'] = params[ptile_dict[key]]
            deciles_forecast[key]['df_loss'] = Loss_Calculator().create_loss_dataframe_region(
                df_train_nora, None, df_predictions, train_period=train_period,
                which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
        return deciles_forecast
