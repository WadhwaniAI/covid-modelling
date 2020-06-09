import pandas as pd
import numpy as np

from data.dataloaders.base import BaseLoader

class JHULoader(BaseLoader):
    def __init__(self):
        super().__init__()

    # helper function for modifying the dataframes such that each row is a snapshot of a country on a particular day
    def _modify_dataframe(self, df, column_name='RecoveredCases'):
        cases_matrix = df.to_numpy()[:, 4:]
        cases_array = cases_matrix.reshape(-1, 1)

        province_info_matrix = df.to_numpy()[:, :4]
        province_info_array = np.repeat(province_info_matrix, cases_matrix.shape[1], axis=0)

        date = pd.to_datetime(df.columns[4:])
        date_array = np.tile(date, cases_matrix.shape[0]).reshape(-1, 1)

        data = np.concatenate((province_info_array, date_array, cases_array), axis=1)
        df = pd.DataFrame(data=data, columns=['Province/State', 'Country/Region', 'Lat', 'Long', 'Date', column_name])
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    # Loads time series case data for every country (and all provinces within certain countries) from JHU's github repo
    def load_data(self):
        """
        This function parses the confirmed, death and recovered CSVs on JHU's github repo and converts them to pandas dataframes
        Columns of returned dataframe : 
        ['Province/State', 'Country/Region', 'Lat', 'Long', 'Date', 'ConfirmedCases', 'Deaths', 'RecoveredCases', 'ActiveCases']
        """
        df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
        df_death = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
        df_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

        df_confirmed = self._modify_dataframe(df_confirmed, 'ConfirmedCases')
        df_death = self._modify_dataframe(df_death, 'Deaths')
        df_recovered = self._modify_dataframe(df_recovered, 'RecoveredCases')

        df_master = df_confirmed.merge(df_death, how='outer').merge(df_recovered, how='outer')
        df_master['ActiveCases'] = df_master['ConfirmedCases'] - df_master['Deaths'] - df_master['RecoveredCases']

        return df_master

    def get_jhu_data(self):
        return self.load_data()
