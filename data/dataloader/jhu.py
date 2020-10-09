import pandas as pd
import numpy as np

import datetime

from data.dataloader.base import BaseLoader

class JHULoader(BaseLoader):
    def __init__(self):
        super().__init__()

    # helper function for modifying the dataframes such that each row is a snapshot of a country on a particular day
    def _modify_dataframe(self, df, column_name='RecoveredCases', province_info_column_idx=4):
        cases_matrix = df.to_numpy()[:, province_info_column_idx:]
        cases_array = cases_matrix.reshape(-1, 1)

        province_info_matrix = df.to_numpy()[:, :province_info_column_idx]
        province_info_array = np.repeat(province_info_matrix, cases_matrix.shape[1], axis=0)
        province_info_columns = df.columns[:province_info_column_idx]

        date = pd.to_datetime(df.columns[province_info_column_idx:])
        date_array = np.tile(date, cases_matrix.shape[0]).reshape(-1, 1)

        data = np.concatenate((province_info_array, date_array, cases_array), axis=1)
        df = pd.DataFrame(data=data, columns=province_info_columns.to_list() + ['Date', column_name])
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    def _load_from_daily_reports(self):
        starting_date = datetime.datetime.strptime('04-12-2020', "%m-%d-%Y")
        main_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/' + \
                   'master/csse_covid_19_data/csse_covid_19_daily_reports_us/{}.csv'

        total_days = (datetime.datetime.today() - starting_date).days
        df_master = pd.read_csv(main_url.format(starting_date.strftime("%m-%d-%Y")))
        df_master['date'] = starting_date
        for i in range(1, total_days+1):
            curr_date = starting_date + datetime.timedelta(days=i)
            try:
                df = pd.read_csv(main_url.format(curr_date.strftime("%m-%d-%Y")))
                df['date'] = curr_date
                df_master = pd.concat([df_master, df], ignore_index=True)
            except Exception as e:
                pass

        return df_master

    def _load_data_from_time_series_us(self):
        """
        This function parses the confirmed, death and recovered CSVs on JHU's 
        github repo and converts them to pandas dataframes
        Columns of returned dataframe : 
        ['Province/State', 'Country/Region', 'Lat', 'Long', 'Date', 
        'ConfirmedCases', 'Deaths', 'RecoveredCases', 'ActiveCases']
        """
        main_url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/' + \
                 'master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{}_US.csv'
        df_confirmed = pd.read_csv(main_url.format('confirmed'))
        df_death = pd.read_csv(main_url.format('deaths'))
        df_confirmed = self._modify_dataframe(df_confirmed, column_name='ConfirmedCases',
                                              province_info_column_idx=11)
        df_death = self._modify_dataframe(df_death, column_name='Deaths',
                                          province_info_column_idx=12)

        df_confirmed['Population'] = df_death['Population']
        df_confirmed['Deaths'] = df_death['Deaths']
        
        return df_confirmed

    # Loads time series case data for every country (and all provinces within certain countries) from JHU's github repo
    def _load_data_from_time_series_global(self):
        """
        This function parses the confirmed, death and recovered CSVs on JHU's 
        github repo and converts them to pandas dataframes
        Columns of returned dataframe : 
        ['Province/State', 'Country/Region', 'Lat', 'Long', 'Date', 
        'ConfirmedCases', 'Deaths', 'RecoveredCases', 'ActiveCases']
        """
        main_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/' + \
                   'master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{}_global.csv'
        df_confirmed = pd.read_csv(main_url.format('confirmed'))
        df_death = pd.read_csv(main_url.format('deaths'))
        df_recovered = pd.read_csv(main_url.format('recovered'))

        df_confirmed = self._modify_dataframe(df_confirmed, 'ConfirmedCases')
        df_death = self._modify_dataframe(df_death, 'Deaths')
        df_recovered = self._modify_dataframe(df_recovered, 'RecoveredCases')

        df_master = df_confirmed.merge(df_death, how='outer').merge(df_recovered, how='outer')
        df_master['ActiveCases'] = df_master['ConfirmedCases'] - df_master['Deaths'] - df_master['RecoveredCases']

        return df_master

    def load_data(self):
        dataframes = {}
        dataframes['df_us_states'] = self._load_from_daily_reports()
        dataframes['df_us_counties'] = self._load_data_from_time_series_us()
        dataframes['df_global'] = self._load_data_from_time_series_global()

        return dataframes

    def get_jhu_data(self):
        return self.load_data()

if __name__ == '__main__':
    obj = JHULoader()
    df_states = obj._load_data_from_time_series_us()
