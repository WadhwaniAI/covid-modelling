import pandas as pd
import numpy as np

import datetime

from data.dataloader.base import BaseLoader

class JHULoader(BaseLoader):
    """Dataloader that outputs time series case data for US states, counties, and US from
        the JHU github repo 'https://www.github.com/CSSEGISandData/COVID-19/'

        Allows the user to do fitting on US states, US counties, and all countries

    Args:
        BaseLoader (abstract class): Abstract Data Loader Class
    """
    def __init__(self):
        super().__init__()

    def _modify_dataframe(self, df, column_name='RecoveredCases', province_info_column_idx=4):
        """Helper function for modifying the dataframes such that each row is a 
            snapshot of a country on a particular day

        Args:
            df (pd.DataFrame): dataframe to be modified
            column_name (str, optional): Modification to be done for which column. Defaults to 'RecoveredCases'.
            province_info_column_idx (int, optional): What is the column index of the 
            province info column. Defaults to 4.

        Returns:
            pd.DataFrame: Modified dataframe
        """
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
        """
        This function parses the CSVs from JHU's daily_reports module 
        and converts them to pandas dataframes
        This returns case counts for all US counties, states, and US as a whole

        Important to note that this returns `deceased`, `total`, `active` and `recovered` numbers

        Returns:
            pd.DataFrame: dataframe of case counts
        """
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
            except Exception:
                pass

        return df_master

    def _load_data_from_time_series_us(self):
        """
        This function parses the confirmed, death and recovered CSVs from JHU's
        time_series_us module and converts them to pandas dataframes
        This returns case counts for all US states

        Important to note that this returns ONLY `deceased` and `total` numbers.
        `active` and `recovered` are not returned.

        Returns:
            pd.DataFrame: dataframe of case counts
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
        This function parses the `confirmed`, `deceased` and `recovered` CSVs from JHU's 
        time_series_global module and converts them to pandas dataframes
        This returns case counts for all countries (including US)

        `active`, `recovered`, `deceased` and `total` are all returned

        Columns of returned dataframe :

        Returns:
            pd.DataFrame: ['Province/State', 'Country/Region', 'Lat', 'Long', 'Date', 
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

    def pull_dataframes(self, **kwargs):
        """Function for pullling dataframes from all modules on JHU's github repo

        Returns:
            dict{str : pd.DatFrame}: Dict of dataframes

            df_us_states : US States, from daily reports
            df_us_counties : US Counties, States, and US, from time series US
            df_global : All countries apart from US, from time series global
        """
        dataframes = {}
        dataframes['df_us_states'] = self._load_from_daily_reports()
        dataframes['df_us_counties'] = self._load_data_from_time_series_us()
        dataframes['df_global'] = self._load_data_from_time_series_global()

        return dataframes

    def pull_dataframes_cached(self, reload_data=False, label=None, **kwargs):
        return super().pull_dataframes_cached(reload_data=reload_data, label=label, **kwargs)

    def get_data(self, dataframe, region, sub_region=None, reload_data=False, **kwargs):
        """Main function serving as handshake between data and fitting modules

        Args:
            dataframe (str): Which df to use for fitting. Can be `global`, `us_states`, `us_counties`.
            region (str): Which region data to do fitting on
            sub_region (str, optional): Which subregion to do fitting on. Defaults to None.
            reload_data (bool, optional): arg for pull_dataframes_cached. If true, data is 
            pulled afresh, rather than using the cache. Defaults to False.

        Raises:
            ValueError: If `dataframe` is `us_counties`, a `sub_region` must be provided. 
            ValueError: `dataframe` can be only 1 of `global`, `us_states`, `us_counties`.

        Returns:
            dict: dict with singular element containing the processed dataframe
        """
        dataframes = self.pull_dataframes_cached(reload_data=reload_data, **kwargs)
        df = dataframes[f'df_{dataframe}']
        if dataframe == 'global':
            df.rename(columns={"ConfirmedCases": "total", "Deaths": "deceased",
                            "RecoveredCases": "recovered", "ActiveCases": "active",
                            "Date": "date"}, inplace=True)
            df.drop(["Lat", "Long"], axis=1, inplace=True)
            df = df[df['Country/Region'] == region]
            if sub_region is None:
                df = df[pd.isna(df['Province/State'])]
            else:
                df = df[df['Province/State'] == sub_region]

        elif dataframe == 'us_states':
            drop_columns = ['Last_Update', 'Lat', 'Long_', 'FIPS', 'Incident_Rate',
                            'People_Hospitalized', 'Mortality_Rate', 'UID', 'ISO3',
                            'Testing_Rate', 'Hospitalization_Rate']
            df.drop(drop_columns, axis=1, inplace=True)
            df.rename(columns={"Confirmed": "total", "Deaths": "deceased",
                            "Recovered": "recovered", "Active": "active",
                            "People_Tested": "tested", "Date": "date"}, inplace=True)
            df = df[['date', 'Province_State', 'Country_Region', 'total', 'active',
                    'recovered', 'deceased', 'tested']]
            df = df[df['Province_State'] == region]

        elif dataframe == 'us_counties':
            drop_columns = ['UID', 'iso2', 'iso3', 'code3', 'FIPS',
                            'Lat', 'Long_']
            df.drop(drop_columns, axis=1, inplace=True)
            df.rename(columns={"ConfirmedCases": "total", "Deaths": "deceased",
                            "Date": "date"}, inplace=True)
            df = df[['date', 'Admin2', 'Province_State', 'Country_Region', 'Combined_Key',
                    'Population', 'total', 'deceased']]
            if sub_region is None:
                raise ValueError(
                    'Please provide a county name ie, the sub_region key')
            df = df[(df['Province_State'] == region)
                    & (df['Admin2'] == sub_region)]

        else:
            raise ValueError('Unknown dataframe type given as input to user')

        df.reset_index(drop=True, inplace=True)
        return {"data_frame": df}
