import pandas as pd
import numpy as np
import datetime
import requests

from data.dataloader.base import BaseLoader

class Covid19IndiaLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def load_data(self):
        """
        This function parses multiple JSONs from covid19india.org
        It then converts the data into pandas dataframes
        It returns the following dataframes as a dict : 
        - df_tested : Time series of people tested in India
        - df_statewise : Today's snapshot of cases in India, statewise
        - df_india_time_series : Time series of cases in India (nationwide)
        - df_districtwise : Today's snapshot of cases in India, districtwise
        - df_raw_data : Patient level information of cases
        - df_deaths_recoveries : Patient level information of deaths and recoveries
        - [NOT UPDATED ANYMORE] df_travel_history : Travel history of some patients (Unofficial : from newsarticles, twitter, etc)
        - df_resources : Repository of testing labs, fundraising orgs, government helplines, etc
        """

        # List of dataframes to return
        dataframes = {}

        # Parse data.json file
        data = requests.get('https://api.covid19india.org/data.json').json()

        # Create dataframe for testing data
        df_tested = pd.DataFrame.from_dict(data['tested'])
        dataframes['df_tested'] = df_tested

        # Create dataframe for statewise data
        df_statewise = pd.DataFrame.from_dict(data['statewise'])
        dataframes['df_statewise'] = df_statewise

        # Create dataframe for time series data
        df_india_time_series = pd.DataFrame.from_dict(data['cases_time_series'])
        df_india_time_series['date'] = pd.to_datetime([datetime.datetime.strptime(
            x[:6] + ' 2020', '%d %b %Y') for x in df_india_time_series['date']])
        dataframes['df_india_time_series'] = df_india_time_series

        # Parse state_district_wise.json file
        data = requests.get(
            'https://api.covid19india.org/state_district_wise.json').json()
        states = data.keys()
        for state in states:
            for district, district_dict in data[state]['districtData'].items():
                delta_dict = dict([('delta_'+k, v)
                                for k, v in district_dict['delta'].items()])
                data[state]['districtData'][district].update(delta_dict)
                del data[state]['districtData'][district]['delta']

        columns = ['state', 'district', 'active', 'confirmed', 'deceased',
                'recovered', 'delta_confirmed', 'delta_deceased', 'delta_recovered']
        df_districtwise = pd.DataFrame(columns=columns)
        for state in states:
            df = pd.DataFrame.from_dict(
                data[state]['districtData']).T.reset_index()
            del df['notes']
            df.columns = columns[1:]
            df['state'] = state
            df = df[columns]
            df_districtwise = pd.concat([df_districtwise, df], ignore_index=True)
        dataframes['df_districtwise'] = df_districtwise

        # Parse raw_data.json file
        # Create dataframe for raw history
        data = requests.get('https://api.covid19india.org/raw_data.json').json()
        df_raw_data_old = pd.DataFrame.from_dict(data['raw_data'])
        dataframes['df_raw_data_old'] = df_raw_data_old

        data = requests.get('https://api.covid19india.org/raw_data1.json').json()
        df_raw_data_1 = pd.DataFrame.from_dict(data['raw_data'])

        data = requests.get('https://api.covid19india.org/raw_data2.json').json()
        df_raw_data_2 = pd.DataFrame.from_dict(data['raw_data'])

        data = requests.get('https://api.covid19india.org/raw_data3.json').json()
        df_raw_data_3 = pd.DataFrame.from_dict(data['raw_data'])

        data = requests.get('https://api.covid19india.org/raw_data4.json').json()
        df_raw_data_4 = pd.DataFrame.from_dict(data['raw_data'])

        data = requests.get('https://api.covid19india.org/raw_data5.json').json()
        df_raw_data_5 = pd.DataFrame.from_dict(data['raw_data'])

        dataframes['df_raw_data'] = pd.concat(
            [df_raw_data_1, df_raw_data_2, df_raw_data_3, df_raw_data_4, df_raw_data_5], ignore_index=True)

        # Parse deaths_recoveries.json file
        data = requests.get('https://api.covid19india.org/deaths_recoveries.json').json()
        df_raw_data_2 = pd.DataFrame.from_dict(data['deaths_recoveries'])
        dataframes['df_deaths_recoveries'] = df_raw_data_2

        data = requests.get('https://api.covid19india.org/districts_daily.json').json()

        df_districts = pd.DataFrame(columns=['notes', 'active', 'confirmed', 'deceased', 
                                             'recovered', 'date', 'state', 'district'])
        for state in data['districtsDaily'].keys():
            for dist in data['districtsDaily'][state].keys():
                df = pd.DataFrame.from_dict(data['districtsDaily'][state][dist])
                df['state'] = state
                df['district'] = dist
                df_districts = pd.concat([df_districts, df], ignore_index=True)

        df_districts = df_districts[['state', 'district', 'date',
                                    'active', 'confirmed', 'deceased', 'recovered', 'notes']]
        df_districts.loc[df_districts['district'] ==
                        'Bengaluru', 'district'] = 'Bengaluru Urban'
        df_districts.loc[df_districts['district'] ==
                        'Ahmadabad', 'district'] = 'Ahmedabad'

        numeric_cols = ['active', 'confirmed', 'deceased', 'recovered']
        df_districts[numeric_cols] = df_districts[numeric_cols].apply(
            pd.to_numeric)
        dataframes['df_districts'] = df_districts

        
        data = requests.get('https://api.covid19india.org/state_test_data.json').json()



        # Parse travel_history.json file
        # Create dataframe for travel history
        """
        !!!! TRAVEL HISTORY HAS BEEN DEPRECATED !!!!
        """
        data = requests.get(
            'https://api.covid19india.org/travel_history.json').json()
        df_travel_history = pd.DataFrame.from_dict(data['travel_history'])
        dataframes['df_travel_history'] = df_travel_history

        data = requests.get(
            'https://api.covid19india.org/resources/resources.json').json()
        df_resources = pd.DataFrame.from_dict(data['resources'])
        dataframes['df_resources'] = df_resources

        return dataframes

    def get_covid19india_api_data(self):
        return self.load_data()
