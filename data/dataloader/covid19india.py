import pandas as pd
import numpy as np
import datetime
import copy
import requests

from data.dataloader.base import BaseLoader

class Covid19IndiaLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def _load_data_json(self, dataframes):
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
        return dataframes

    def _load_state_district_wise_json(self, dataframes):
        # Parse state_district_wise.json file
        data = requests.get('https://api.covid19india.org/state_district_wise.json').json()
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

        data = requests.get('https://api.covid19india.org/state_district_wise.json').json()
        df_statecode = pd.DataFrame.from_dict(data)
        df_statecode = df_statecode.drop(['districtData']).T
        statecode_to_state_dict = dict(zip(df_statecode['statecode'], df_statecode.index))

        return dataframes, statecode_to_state_dict

    def _load_raw_data_json(self, dataframes):
         # Parse raw_data.json file
        raw_data_dataframes = []
        for i in range(1, 21):
            try:
                data = requests.get(f'https://api.covid19india.org/raw_data{i}.json').json()
                raw_data_dataframes.append(pd.DataFrame.from_dict(data['raw_data']))
            except Exception as e:
                break

        dataframes['df_raw_data'] = pd.concat(raw_data_dataframes, ignore_index=True)
        return dataframes

    def _load_districts_daily_json(self, dataframes):
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

        numeric_cols = ['active', 'confirmed', 'deceased', 'recovered']
        df_districts[numeric_cols] = df_districts[numeric_cols].apply(
            pd.to_numeric)
        dataframes['df_districts'] = df_districts
        return dataframes

    def _load_data_all_json_district(self, dataframes, statecode_to_state_dict):
        data = requests.get('https://api.covid19india.org/v4/data-all.json').json()

        for date in data.keys():
            date_dict = data[date]
            # Remove all the states which don't have district data in them
            date_dict = {state : state_dict for state, state_dict in date_dict.items() \
                if 'districts' in state_dict.keys()}
            data[date] = date_dict
            
        # Remove all the dates which have 0 states with district data after pruning
        data = {date : date_dict for date, date_dict in data.items() if len(date_dict) > 0}

        # Make the districts key data the only data available for the state key
        for date in data.keys():
            for state in data[date].keys():
                # Make the districts key dict the main dict itself for a particular date, state
                data[date][state] = data[date][state]['districts']
                state_dict = data[date][state]
                # Keep only those district dicts for which cumulative data (total key) is available
                state_dict = {dist : dist_dict for dist, dist_dict in state_dict.items() \
                    if 'total' in dist_dict.keys()}
                data[date][state] = state_dict

                # Make the total key dict the main dict itself for a particular date, state, dist
                for district in data[date][state].keys():
                        data[date][state][district] = data[date][state][district]['total']
                
                # For a particular date, state, dist, only keep those keys for which have confirmed, recovered, deceased are all available
                state_dict = {dist: dist_dict for dist, dist_dict in state_dict.items() \
                    if {'confirmed', 'recovered', 'deceased'} <= dist_dict.keys()}
                data[date][state] = state_dict
            
            # Remove all the states for a particular date which don't have district that satisfied above criteria
            date_dict = data[date]
            date_dict = {state : state_dict for state, state_dict in date_dict.items() if len(state_dict) > 0}
            data[date] = date_dict
            
        # Remove all the dates which have 0 states with district data after pruning
        data = {date : date_dict for date, date_dict in data.items() if len(date_dict) > 0}

        df_districts_all = pd.DataFrame(columns=['date', 'state', 'district', 'confirmed', 'active', 
                                                 'recovered', 'deceased', 'tested', 'migrated'])
        for date in data.keys():
            for state in data[date].keys():
                df_date_state = pd.DataFrame.from_dict(data[date][state]).T.reset_index()
                df_date_state = df_date_state.rename({'index' : 'district'}, axis='columns')
                df_date_state['active'] = df_date_state['confirmed'] - \
                    (df_date_state['recovered'] + df_date_state['deceased'])
                df_date_state['state'] = statecode_to_state_dict[state]
                df_date_state['date'] = date
                df_districts_all = pd.concat([df_districts_all, df_date_state], ignore_index=True)

        numeric_cols = ['confirmed', 'active', 'recovered', 'deceased', 'tested', 'migrated']
        df_districts_all.loc[:, numeric_cols] = df_districts_all.loc[:, numeric_cols].apply(pd.to_numeric)
        dataframes['df_districts_all'] = df_districts_all
        return dataframes

    def _load_data_all_json_state(self, dataframes, statecode_to_state_dict):
        data = requests.get('https://api.covid19india.org/v4/data-all.json').json()
        for date in data.keys():
            date_dict = data[date]
            # Remove all the states which don't have district data in them
            date_dict = {state : state_dict for state, state_dict in date_dict.items() if 'districts' in state_dict.keys()}
            data[date] = date_dict
            
        # Remove all the dates which have 0 states with district data after pruning
        data = {date : date_dict for date, date_dict in data.items() if len(date_dict) > 0}

        # Make the districts key data the only data available for the state key
        for date in data.keys():
            for state in data[date].keys():
                # Make the districts key dict the main dict itself for a particular date, state
                data[date][state] = data[date][state]['total']
                
            date_dict = {state: state_dict for state, state_dict in data[date].items() \
                        if {'confirmed', 'recovered', 'deceased'} <= state_dict.keys()}
            data[date] = date_dict
            
        # Remove all the dates which have 0 states with district data after pruning
        data = {date : date_dict for date, date_dict in data.items() if len(date_dict) > 0}

        df_states_all = pd.DataFrame(columns=['date', 'state', 'confirmed', 'active', 'recovered', 'deceased', 'tested', 'migrated'])
        for date in data.keys():
            df_date = pd.DataFrame.from_dict(data[date]).T.reset_index()
            df_date = df_date.rename({'index' : 'state'}, axis='columns')
            df_date['active'] = df_date['confirmed'] - (df_date['recovered'] + df_date['deceased'])
            df_date['state'] = pd.Series([statecode_to_state_dict[state_code] for state_code in df_date['state']])
            df_date['date'] = date
            df_states_all = pd.concat([df_states_all, df_date], ignore_index=True)
        
        numeric_cols = ['confirmed', 'active', 'recovered', 'deceased', 'tested', 'migrated']
        df_states_all.loc[:, numeric_cols] = df_states_all.loc[:, numeric_cols].apply(pd.to_numeric)
        dataframes['df_states_all'] = df_states_all
        return dataframes

    def load_data(self, load_raw_data=False, load_districts_daily=False):
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

        dataframes = self._load_data_json(dataframes)
        dataframes, statecode_to_state_dict = self._load_state_district_wise_json(dataframes)
        if load_raw_data:
            dataframes = self._load_raw_data_json(dataframes)
        if load_districts_daily:
            dataframes = self._load_districts_daily_json(dataframes)
        dataframes = self._load_data_all_json_district(dataframes, statecode_to_state_dict)
        dataframes = self._load_data_all_json_state(dataframes, statecode_to_state_dict)

        return dataframes

    def get_covid19india_api_data(self):
        return self.load_data()
