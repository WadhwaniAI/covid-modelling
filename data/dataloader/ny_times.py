import pandas as pd
import numpy as np
from data.dataloader.base import BaseLoader


class NYTLoader(BaseLoader):
    """Dataloader that time series case data for US states, counties, and US from
        the NY Times github repo 'https://www.github.com/nytimes/covid-19-data/'

        Allows the user to do fitting on US states, US counties

    Args:
        BaseLoader (abstract class): Abstract Data Loader Class
    """
    def __init__(self):
        super().__init__()

    # Loads time series case data for US counties and states from the New York Times github repo
    def pull_dataframes(self, **kwargs):
        """
        This function parses the us-states and us-counties CSVs on
        NY Times's github repo and converts them to pandas dataframes
        Returns dict of dataframes for states and counties

        Returns:
            dict{str : pd.DataFrame}: dict of pulled dataframes
        """
        dataframes = dict()
        df_counties = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv',
                                  error_bad_lines=False)
        df_states = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv',
                                error_bad_lines=False)
        dataframes['counties'] = df_counties
        dataframes['states'] = df_states
        return dataframes

    def pull_dataframes_cached(self, reload_data=False, label=None, **kwargs):
        return super().pull_dataframes_cached(reload_data=reload_data, label=label, **kwargs)

    def get_data(self, state, county=None, reload_data=False, **kwargs):
        """Main function serving as handshake between data and fitting modules

        Args:
            state (str): The US state of the region of interest
            county (str, optional): The county of the region of interest. 
            If None, fitting done on given state. Defaults to None.
            reload_data (bool, optional): arg for pull_dataframes_cached. If true, data is 
            pulled afresh, rather than using the cache. Defaults to False.

        Returns:
            dict: dict with singular element containing the processed dataframe
        """
        dataframes = self.pull_dataframes_cached(reload_data=reload_data, **kwargs)
        if county is not None:
            df = dataframes['counties']
            df = df[np.logical_and(df['state'] == state, df['county'] == county)]
        else:
            df = dataframes['states']
            df = df[df['state'] == state]
        df.loc[:, 'date'] = pd.to_datetime(df['date'])
        df.rename(columns={"cases": "total", "deaths": "deceased"}, inplace=True)
        df.drop('fips', axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return {"data_frame": df}
