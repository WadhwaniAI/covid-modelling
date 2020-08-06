import pandas as pd

from data.dataloader.base import BaseLoader


class CovidTrackingLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    # Loads time series case data for US states from the Covid Tracking API 'https://covidtracking.com/data/api'
    def load_data(self):
        """
        This function parses the us-states and us-counties CSVs on NY Times's github repo and converts them to pandas dataframes
        Returns dict of dataframes for states and counties
        """
        dataframes = dict()
        df_states = pd.read_csv('https://covidtracking.com/api/v1/states/daily.csv')
        dataframes['states'] = df_states
        return dataframes

    def get_covid_tracking_data(self):
        return self.load_data()
