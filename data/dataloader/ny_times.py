import pandas as pd

from data.dataloader.base import BaseLoader


class NYTLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    # Loads time series case data for US counties and states from the New York Times github repo
    def load_data(self):
        """
        This function parses the us-states and us-counties CSVs on NY Times's github repo and converts them to pandas dataframes
        Returns dict of dataframes for states and counties
        """
        dataframes = dict()
        df_counties = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv',
                                  error_bad_lines=False)
        df_states = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv',
                                error_bad_lines=False)
        dataframes['counties'] = df_counties
        dataframes['states'] = df_states
        return dataframes

    def get_jhu_data(self):
        return self.load_data()
