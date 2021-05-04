import pandas as pd
import datetime
import copy
import requests

from data.dataloader.base import BaseLoader

class Covid19IndiaLoader(BaseLoader):
    """Dataloader that gets casecount data from 'https://api.covid19india.org'
    We use the JSON api and not the CSV api

    Different API are accessed and then converted into pd.DataFrames

    Full list of dataframes are given in the docstrings of pull_dataframes

    Args:
        BaseLoader (abstract class): Abstract Data Loader Class
    """
    def __init__(self):
        super().__init__()

    def _load_data_json(self):
        """Returns dataframes from data.json

        df_tested : dataframe of testing data
        df_statewise : dataframe of statewise data (today's snapshot)
        df_india_time_series : dataframe of india cases (time series)

        Returns:
            [pd.DataFrame]: list of dataframes
        """
        # Parse data.json file
        data = requests.get('https://api.covid19india.org/data.json').json()

        # Create dataframe for testing data
        df_tested = pd.DataFrame.from_dict(data['tested'])

        # Create dataframe for statewise data
        df_statewise = pd.DataFrame.from_dict(data['statewise'])

        # Create dataframe for time series data
        df_india_time_series = pd.DataFrame.from_dict(data['cases_time_series'])
        df_india_time_series['date'] = pd.to_datetime([datetime.datetime.strptime(
            x.split(' ')[0] + ' ' + x.split(' ')[1][:3] + ' 2020', '%d %b %Y') for x in
            df_india_time_series['date']])

        return df_tested, df_statewise, df_india_time_series

    def _load_state_district_wise_json(self):
        """Loads dataframes from the state_district_wise.json file

        df_districtwise : Today's snapshot of district-wise cases
        statecode_to_state_dict : Mapping statecode to state name

        Returns:
            pd.DataFrame, dict: df_districtwise, statecode_to_state_dict
        """
        # Load state_district_wise.json file
        data = requests.get('https://api.covid19india.org/state_district_wise.json').json()

        # Create statecode_to_state_dict
        df_statecode = pd.DataFrame.from_dict(data)
        df_statecode = df_statecode.drop(['districtData']).T
        statecode_to_state_dict = dict(
            zip(df_statecode['statecode'], df_statecode.index))

        # Create df_districtwise
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

        return df_districtwise, statecode_to_state_dict

    def _load_raw_data_json(self, NUM_RAW_DFS=30):
        """Loads raw_data from raw_data{i}.json

        df_raw : patient level information

        Args:
            NUM_RAW_DFS (int, optional): Number of raw data json files to consider. Defaults to 30.
        """
        # Parse raw_data.json file
        raw_data_dataframes = []
        for i in range(1, NUM_RAW_DFS+1):
            try:
                data = requests.get(f'https://api.covid19india.org/raw_data{i}.json').json()
                raw_data_dataframes.append(pd.DataFrame.from_dict(data['raw_data']))
            except Exception:
                break

        df_raw = pd.concat(raw_data_dataframes, ignore_index=True)
        
        return df_raw

    def _load_districts_daily_json(self):
        """Loads history of cases district wise from districts_daily.json

        Returns:
            pd.DataFrame: df_districts
        """
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
        return df_districts

    def _load_districts_csv(self):
        df = pd.read_csv('https://api.covid19india.org/csv/latest/districts.csv')
        df.columns = [x.lower() for x in df.columns]
        df['active'] = df['confirmed'] - (df['recovered'] + df['deceased'])
        numeric_cols = ['confirmed', 'active', 'recovered', 'deceased', 'tested', 'other']
        df.loc[:, numeric_cols] = df.loc[:, numeric_cols].apply(pd.to_numeric)
        df = df.fillna(0)
        # df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
        return df

    def _load_data_all_json_district(self, statecode_to_state_dict):
        """Loads history of cases district wise from data-all.json

        Args:
            statecode_to_state_dict (dict): dict mapping state code to state name

        Returns:
            pd.DataFrame: df_districts_all
        """
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
                df_districts_all = pd.concat([df_districts_all, df_date_state], ignore_index=True, sort=False)

        numeric_cols = ['confirmed', 'active', 'recovered', 'deceased', 'tested', 'migrated']
        df_districts_all.loc[:, numeric_cols] = df_districts_all.loc[:, numeric_cols].apply(pd.to_numeric)
        
        return df_districts_all

    def _load_data_all_json_state(self, statecode_to_state_dict):
        """Loads history of cases state wise from data-all.json

        Args:
            statecode_to_state_dict (dict): dict mapping state code to state name

        Returns:
            pd.DataFrame: df_state_all
        """
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
        data = {date: date_dict for date, date_dict in data.items() if len(date_dict) > 0}

        df_states_all = pd.DataFrame(columns=['date', 'state', 'confirmed', 'active', 'recovered', 'deceased', 'tested', 'migrated'])
        for date in data.keys():
            df_date = pd.DataFrame.from_dict(data[date]).T.reset_index()
            df_date = df_date.rename({'index': 'state'}, axis='columns')
            df_date['active'] = df_date['confirmed'] - (df_date['recovered'] + df_date['deceased'])
            df_date['state'] = pd.Series([statecode_to_state_dict[state_code] for state_code in df_date['state']])
            df_date['date'] = date
            df_states_all = pd.concat([df_states_all, df_date], ignore_index=True)
        
        numeric_cols = ['confirmed', 'active', 'recovered', 'deceased', 'tested', 'migrated']
        df_states_all.loc[:, numeric_cols] = df_states_all.loc[:, numeric_cols].apply(pd.to_numeric)
        return df_states_all

    def pull_dataframes(self, load_raw_data=False, load_districts_daily=False, **kwargs):
        """
        This function parses multiple JSONs from covid19india.org
        It then converts the data into pandas dataframes
        It returns the following dataframes as a dict : 
        - df_tested : Time series of people tested in India
        - df_statewise : Today's snapshot of cases in India, statewise
        - df_india_time_series : Time series of cases in India (nationwide)
        - df_districtwise : Today's snapshot of cases in India, districtwise
        - df_raw_data : Patient level information of cases
        - df_districts_daily : History of cases district wise obtained from districts_daily.json
        - df_districts_all : History of cases district wise obtained from data_all.json
        - df_states_all : History of cases state wise obtained from data_all.json
        """

        # List of dataframes to return
        dataframes = {}

        df_tested, df_statewise, df_india_time_series = self._load_data_json()
        dataframes['df_tested'] = df_tested
        dataframes['df_statewise'] = df_statewise
        dataframes['df_india_time_series'] = df_india_time_series
        # df_districtwise, statecode_to_state_dict = self._load_state_district_wise_json()
        # dataframes['df_districtwise'] = df_districtwise
        if load_raw_data:
            df_raw = self._load_raw_data_json()
            dataframes['df_raw_data'] = df_raw
        if load_districts_daily:
            df_districts = self._load_districts_daily_json()
            dataframes['df_districts_daily'] = df_districts
        # df_districts_all = self._load_data_all_json_district(statecode_to_state_dict)
        # dataframes['df_districts_all'] = df_districts_all
        # df_states_all = self._load_data_all_json_state(statecode_to_state_dict)
        # dataframes['df_states_all'] = df_states_all
        df_districts = self._load_districts_csv()
        dataframes['df_districts'] = df_districts

        return dataframes

    def pull_dataframes_cached(self, reload_data=False, label=None, **kwargs):
        return super().pull_dataframes_cached(reload_data=reload_data, label=label, **kwargs)

    def get_data(self, state='Maharashtra', district='Mumbai', use_dataframe='data_all',
                 reload_data=False, **kwargs):
        """Main function serving as handshake between data and fitting modules

        Args:
            state (str, optional): State to fit on. Defaults to 'Maharashtra'.
            district (str, optional): District to fit on. If given, get_data_district is called. 
            Else, get_data_state is called. Defaults to 'Mumbai'.
            use_dataframe (str, optional): Which dataframe to use for districts.
             Can be data_all/districts_daily. Defaults to 'data_all'.
            reload_data (bool, optional): arg for pull_dataframes_cached. If true, data is 
            pulled afresh, rather than using the cache. Defaults to False.

        Returns:
            dict { str : pd.DataFrame }  : Processed dataframe
        """
        if not district is None:
            return {"data_frame": self.get_data_district(state, district, use_dataframe, 
                                                         reload_data, **kwargs)}
        else:
            return {"data_frame": self.get_data_state(state, reload_data, **kwargs)}


    def get_data_state(self, state='Delhi', reload_data=False, **kwargs):
        """Helper function for get_data. Returns state data

        Args:
            state (str, optional): State to fit on. Defaults to 'Delhi'.
            reload_data (bool, optional): arg for pull_dataframes_cached. If true, data is 
            pulled afresh, rather than using the cache. Defaults to False.

        Returns:
            dict { str : pd.DataFrame }  : Processed dataframe
        """
        dataframes = self.pull_dataframes_cached(reload_data=reload_data, **kwargs)
        df_states = copy.copy(dataframes['df_states_all'])
        df_state = df_states[df_states['state'] == state]
        df_state['date'] = pd.to_datetime(df_state['date'])
        df_state = df_state.rename({'confirmed': 'total'}, axis='columns')
        df_state.reset_index(inplace=True, drop=True)
        return df_state

    def get_data_district(self, state='Karnataka', district='Bengaluru', 
                          use_dataframe='data_all', reload_data=False, **kwargs):
        """Helper function for get_data. Returns district data

        Args:
            state (str, optional): State to fit on. Defaults to 'Karnataka'.
            district (str, optional): District to fit on. Defaults to 'Bengaluru'.
            use_dataframe (str, optional) : Which dataframe to use. Can be `data_all`/`districts_daily`.
            reload_data (bool, optional): arg for pull_dataframes_cached. If true, data is
            pulled afresh, rather than using the cache. Defaults to False.

        Returns:
            dict { str : pd.DataFrame }  : Processed dataframe
        """
        dataframes = self.pull_dataframes_cached(reload_data=reload_data, **kwargs)

        if use_dataframe == 'data_all':
            df_districts = copy.copy(dataframes['df_districts_all'])
            df_district = df_districts.loc[(df_districts['state'] == state) & (
                df_districts['district'] == district)]
            df_district.loc[:, 'date'] = pd.to_datetime(df_district.loc[:, 'date'])
            df_district = df_district.rename({'confirmed': 'total'}, axis='columns')
            del df_district['migrated']
            df_district.reset_index(inplace=True, drop=True)
            return df_district

        if use_dataframe == 'districts_daily':
            df_districts = copy.copy(dataframes['df_districts_daily'])
            df_district = df_districts.loc[(df_districts['state'] == state) & (
                df_districts['district'] == district)]
            del df_district['notes']
            df_district.loc[:, 'date'] = pd.to_datetime(df_district.loc[:, 'date'])
            df_district = df_district.loc[df_district['date'] >= '2020-04-24', :]
            df_district = df_district.rename({'confirmed': 'total'}, axis='columns')
            df_district.reset_index(inplace=True, drop=True)
            return df_district

        if use_dataframe == 'districts':
            df_districts = copy.copy(dataframes['df_districts'])
            df_district = df_districts.loc[(df_districts['state'] == state) & (
                    df_districts['district'] == district)]
            df_district.loc[:, 'date'] = pd.to_datetime(df_district.loc[:, 'date'])
            df_district = df_district.loc[df_district['date'] >= '2020-04-24', :]
            df_district = df_district.rename({'confirmed': 'total'}, axis='columns')
            df_district.reset_index(inplace=True, drop=True)
            return df_district
