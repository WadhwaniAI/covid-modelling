import pandas as pd

from data.dataloader.base import BaseLoader


class CovidTrackingLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    # Loads time series case data for US states from the Covid Tracking API 'https://covidtracking.com/data/api'
    def pull_dataframes(self):
        """
        This function parses the us-states and us-counties CSVs on NY Times's github repo and converts them to pandas dataframes
        Returns dict of dataframes for states and counties
        """
        dataframes = dict()
        df_states = pd.read_csv('https://covidtracking.com/api/v1/states/daily.csv')
        df_states_metadata = pd.read_csv('https://api.covidtracking.com/v1/states/info.csv')

        state_code_to_name = dict(zip(df_states_metadata['state'].to_numpy(), 
                                      df_states_metadata['name'].to_numpy()))

        df_states['state_name'] = df_states['state'].apply(lambda x : state_code_to_name[x])
        df_states['date'] = pd.to_datetime(df_states['date'].apply(lambda x: str(x)), format="%Y%m%d")
        df_states = df_states.sort_values(['date', 'state_name'])
        df_states.reset_index(inplace=True, drop=True)
        df_states['active'] = df_states['positive'] - \
            df_states['recovered'] - df_states['death']

        dataframes['df_states'] = df_states
        dataframes['df_states_metadata'] = df_states_metadata
        return dataframes

    def pull_dataframes_cached(self, reload_data=False, label=None, **kwargs):
        return super().pull_dataframes_cached(reload_data=reload_data, label=label, **kwargs)
