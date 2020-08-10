import pandas as pd
import numpy as np
import copy
import os
import pickle
import datetime
from collections import defaultdict

from data.dataloader import Covid19IndiaLoader, RootnetLoader, AthenaLoader


def get_dataframes_cached(loader_class=Covid19IndiaLoader, reload_data=False):
    if loader_class == Covid19IndiaLoader:
        loader_key = 'tracker'
    if loader_class == AthenaLoader:
        loader_key = 'athena'
    os.makedirs("../../misc/cache/", exist_ok=True)
    picklefn = "../../misc/cache/dataframes_ts_{today}_{loader_key}.pkl".format(
        today=datetime.datetime.today().strftime("%d%m%Y"), loader_key=loader_key)
    if reload_data:
        print("pulling from source")
        loader = loader_class()
        dataframes = loader.load_data()
    else:
        try:
            with open(picklefn, 'rb') as pickle_file:
                dataframes = pickle.load(pickle_file)
            print(f'loading from {picklefn}')
        except:
            print("pulling from source")
            loader = loader_class()
            dataframes = loader.load_data()
            with open(picklefn, 'wb+') as pickle_file:
                pickle.dump(dataframes, pickle_file)
    return dataframes


def get_data(state=None, district=None, use_dataframe='data_all', disable_tracker=False,
             filename=None, data_format='new'):
    """Handshake between data module and training module. Returns a dataframe of cases for a particular district/state
       from multiple sources

    If : 
    state, dist are given, use_dataframe == 'districts_daily' : data loaded from covid19india tracker (districts_daily.json)
    state, dist are given, use_dataframe == 'raw_data' : data loaded from covid19india tracker (raw_data.json)
    dist given, state == None : state data loaded from rootnet tracker
    disable_tracker=True, filename != None : data loaded from file (csv file)
        data_format == new  : The new format used by Jerome/Vasudha
        data_format == old  : The old format Puskar/Keshav used to supply data in
    disable_tracker=True, filename == None : data loaded from AWS Athena Database

    Keyword Arguments:
        dataframes {dict} -- dict of dataframes returned from the get_covid19india_api_data function (default: {None})
        state {str} -- Name of state for which data to be loaded (in title case) (default: {None})
        district {str} -- Name of district for which data to be loaded (in title case) (default: {None})
        use_dataframe {str} -- If covid19india tracker being used, what json to use (default: {'districts_daily'})
        disable_tracker {bool} -- Flag to not use tracker (default: {False})
        filename {str} -- Path to CSV file with data (only if disable_tracker == True) (default: {None})
        data_format {str} -- Format of the CSV file (default: {'new'})

    Returns:
        pd.DataFrame -- dataframe of cases for a particular state, district with 4 columns : 
        ['total', 'active', 'deceased', 'recovered']
        (All columns are populated except using raw_data.json)
       
    """
    if disable_tracker:
        if filename != None:
            df_result = get_custom_data_from_file(
                filename, data_format=data_format)
        else:
            df_result = get_custom_data_from_db(state, district)
    elif district != None:
        df_result = get_district_time_series(state=state, district=district, use_dataframe=use_dataframe)
    else:
        df_result = get_state_time_series(state=state)
    return df_result

def get_custom_data_from_db(state='Maharashtra', district='Mumbai'):
    print('fetching from athenadb...')
    dataframes = get_dataframes_cached(loader_class=AthenaLoader)
    df_result = copy.copy(dataframes['new_covid_case_summary'])
    df_result['state'] = 'maharashtra'
    df_result = df_result[np.logical_and(
        df_result['state'] == state.lower(), df_result['district'] == district.lower())]
    df_result = df_result.loc[:, :'deceased']
    df_result.dropna(axis=0, how='any', inplace=True)
    df_result.loc[:, 'date'] = pd.to_datetime(df_result['date'])
    df_result.reset_index(inplace=True, drop=True)
    for col in df_result.columns:
        if col in ['active', 'total', 'recovered', 'deceased']:
            df_result[col] = df_result[col].astype('int64')
    return df_result
    
#TODO add support of adding 0s column for the ones which don't exist
def get_custom_data_from_file(filename, data_format='new'):
    if data_format == 'new':
        df_result = pd.read_csv(filename) 
        df_result = df_result.drop(['Ward/block name', 'Ward number (if applicable)', 'Mild cases (isolated)',
                                    'Moderate cases (hospitalized)', 'Severe cases (In ICU)', 
                                    'Critical cases (ventilated patients)'], axis=1)
        df_result.columns = ['state', 'district', 'date', 'total', 'active', 'recovered', 'deceased']
        df_result.drop(np.arange(3), inplace=True)
        df_result['date'] = pd.to_datetime(df_result['date'], format='%m-%d-%Y')
        df_result = df_result.dropna(subset=['state'], how='any')
        df_result.reset_index(inplace=True, drop=True)
        df_result.loc[:, ['total', 'active', 'recovered', 'deceased']] = df_result[[
            'total', 'active', 'recovered', 'deceased']].apply(pd.to_numeric)
        df_result = df_result[['date', 'state', 'district', 'total', 'active', 'recovered', 'deceased']]
        df_result = df_result.dropna(subset=['date'], how='all')
        
    if data_format == 'old':
        df_result = pd.read_csv(filename)
        df_result['date'] = pd.to_datetime(df_result['date'])
        df_result.columns = [x if x != 'active' else 'active' for x in df_result.columns]
        df_result.columns = [x if x != 'confirmed' else 'total' for x in df_result.columns]
        
    return df_result
        
def get_state_time_series(state='Delhi'):
    dataframes = get_dataframes_cached()
    df_states = copy.copy(dataframes['df_states_all'])
    df_state = df_states[df_states['state'] == state]
    df_state['date'] = pd.to_datetime(df_state['date'])
    df_state = df_state.rename({'confirmed': 'total'}, axis='columns')
    df_state.reset_index(inplace=True, drop=True)
    return df_state

def get_district_time_series(state='Karnataka', district='Bengaluru', use_dataframe='raw_data'):
    dataframes = get_dataframes_cached()

    if use_dataframe == 'data_all':
        df_districts = copy.copy(dataframes['df_districts_all'])
        df_district = df_districts.loc[(df_districts['state'] == state) & (df_districts['district'] == district)]
        df_district.loc[:, 'date'] = pd.to_datetime(df_district.loc[:, 'date'])
        df_district = df_district.rename({'confirmed': 'total'}, axis='columns')
        df_district.reset_index(inplace=True, drop=True)
        return df_district
    
    if use_dataframe == 'districts_daily':
        df_districts = copy.copy(dataframes['df_districts'])
        df_district = df_districts.loc[(df_districts['state'] == state) & (df_districts['district'] == district)]
        del df_district['notes']
        df_district.loc[:, 'date'] = pd.to_datetime(df_district.loc[:, 'date'])
        df_district = df_district.loc[df_district['date'] >= '2020-04-24', :]
        df_district = df_district.rename({'confirmed': 'total'}, axis='columns')
        df_district.reset_index(inplace=True, drop=True)
        return df_district

    if use_dataframe == 'raw_data':
        if type(dataframes) is dict:
            df_raw_data_1 = copy.copy(dataframes['df_raw_data'])
        else:
            df_raw_data_1 = copy.copy(dataframes)
        if state != None:
            df_raw_data_1 = df_raw_data_1[df_raw_data_1['detectedstate'] == state]
        if district != None:
            df_raw_data_1 = df_raw_data_1[df_raw_data_1['detecteddistrict'] == district]

        if len(df_raw_data_1) == 0:
            out = pd.DataFrame({
                'total':pd.Series([], dtype='int'), 
                'active':pd.Series([], dtype='int'), 
                'deceased':pd.Series([], dtype='int'), 
                'recovered':pd.Series([], dtype='int'),
                'district':pd.Series([], dtype='object'),
                'state':pd.Series([], dtype='object'),
                })
            out.index.name = 'date'
            return out.reset_index()
        
        df_raw_data_1['dateannounced'] = pd.to_datetime(df_raw_data_1['dateannounced'], format='%d/%m/%Y')

        index = pd.date_range(np.min(df_raw_data_1['dateannounced']), np.max(df_raw_data_1['dateannounced']))

        df_district = pd.DataFrame(columns=['total'], index=index)
        df_district['total'] = [0]*len(index)
        for _, row in df_raw_data_1.iterrows():
            try:
                df_district.loc[row['dateannounced']:, 'total'] += 1*int(row['numcases'])
            except Exception:
                df_district.loc[row['dateannounced']:, 'total'] += 1

        df_district.reset_index(inplace=True)
        df_district.columns = ['date', 'total']
        df_district['active'] = [0]*len(df_district)
        df_district['deceased'] = [0]*len(df_district)
        df_district['recovered'] = [0]*len(df_district)
        df_district['district'] = district
        df_district['state'] = state
        return df_district
    
    if use_dataframe == 'deaths_recovs':
        df_deaths_recoveries = copy.copy(dataframes['df_deaths_recoveries'])
        df_deaths_recoveries = df_deaths_recoveries[['date', 'district', 'state', 'patientstatus']]
        df_deaths_recoveries = df_deaths_recoveries[df_deaths_recoveries['state'] == state]
        unknown = df_deaths_recoveries[np.logical_and(df_deaths_recoveries['state'] == state, df_deaths_recoveries['district'] == '')]
        unknown_count = len(unknown[unknown['patientstatus'] == 'Deceased']), len(unknown[unknown['patientstatus'] == 'Recovered'])
        # df_deaths_recoveries.loc[unknown.index, 'district'] = district
        print(f'{unknown_count[0]} deaths and {unknown_count[1]} recoveries in {state} with unknown district')
        df_deaths_recoveries = df_deaths_recoveries[np.logical_and(df_deaths_recoveries['state'] == state, df_deaths_recoveries['district'] == district)]
        if len(df_deaths_recoveries) == 0:
            out = pd.DataFrame({
                'deceased':pd.Series([], dtype='int'), 
                'recovered':pd.Series([], dtype='int'),
                'district':pd.Series([], dtype='object'),
                'state':pd.Series([], dtype='object'),
                })
            out.index.name = 'date'
            return out.reset_index()
        df_deaths_recoveries['date'] = pd.to_datetime(df_deaths_recoveries['date'], format='%d/%m/%Y')
        index = pd.date_range(np.min(df_deaths_recoveries['date']), np.max(df_deaths_recoveries['date']))
        out = pd.DataFrame(0, columns=['deceased', 'recovered'], index=index)
        for _, row in df_deaths_recoveries.iterrows():
            if row['patientstatus'] == 'Deceased':
                out.loc[row['date']:, 'deceased'] += 1
            if row['patientstatus'] == 'Recovered':
                out.loc[row['date']:, 'recovered'] += 1
        out['district'] = district
        out['state'] = state
        out.index.name = 'date'
        return out.reset_index()

def get_district_timeseries_cached(district, state, disable_tracker=False, filename=None, data_format='new'):
    picklefn = "../../cache/{district}_ts_{src}_{today}.pkl".format(
        district=district, today=datetime.datetime.today().strftime("%d%m%Y"), 
        src='athena' if disable_tracker else 'tracker'
    )
    try:
        print(picklefn)
        with open(picklefn, 'rb') as pickle_file:
            district_timeseries = pickle.load(pickle_file)
    except:
        if not disable_tracker:
            if district == 'Bengaluru':
                district = ['Bengaluru Urban', 'Bengaluru Rural']
            elif district == 'Delhi':
                district = ['East Delhi', 'New Delhi', 'North Delhi', 
                'North East Delhi','North West Delhi', 'South Delhi', 
                'South West Delhi','West Delhi', 'Unknown', 'Central Delhi', 
                'Shahdara','South East Delhi','']
        dataframes = get_dataframes_cached()
        district_timeseries = get_data(dataframes, state=state, 
            district=district, disable_tracker=disable_tracker, 
            filename=filename, data_format=data_format)
        with open(picklefn, 'wb+') as pickle_file:
            pickle.dump(district_timeseries, pickle_file)
    return district_timeseries
    
def combine_districts(dfs, new_district, new_state=None):
    datecol, statecol, districtcol = 'date', 'state', 'district'
    new_state = new_state if new_state else dfs[0][statecol].unique()[0]
    out = None
    for i, df in enumerate(dfs):
        df[datecol] = pd.to_datetime(df[datecol])
        df = df.set_index(datecol)
        if statecol in df.columns:
            assert len(df[statecol].unique()) == 1, "max 1 state per df" # only 1 state in df
            df = df.drop(statecol, axis=1)
        if districtcol in df.columns:
            assert len(df[districtcol].unique()) == 1, "max 1 district per df" # only 1 district in each df
            df = df.drop(districtcol, axis=1)
        out = df if i == 0 else out.add(df, fill_value=0)
    
    out[statecol], out[districtcol] = new_state, new_district
    out.index.name = datecol
    return out.reset_index()

def checks(dfs, uniform=True):
    statecol, districtcol = 'state', 'district'
    state_val, district_val = None, None
    for df in dfs:
        if statecol in df.columns:
            # check that all dfs represent only 1 location
            assert len(df[statecol].unique()) == 1, "max 1 state per df"
            if uniform:
                # check that all dfs are of the same district/state
                if state_val == None:
                    state_val = df[statecol].unique()[0]
                assert state_val == df[statecol].unique()[0], "all dfs must have same state"
        if districtcol in df.columns:
            assert len(df[districtcol].unique()) == 1, "max 1 district per df"
            if uniform:
                if district_val == None:
                    district_val = df[districtcol].unique()[0]
                assert district_val == df[districtcol].unique()[0], "all dfs must have same district"
    return state_val, district_val

def concat_sources(from_df_raw_data, from_df_deaths_recoveries, from_df_districtwise):
    datecol, statecol, districtcol = 'date', 'state', 'district'
    state_val, district_val = checks([from_df_raw_data, from_df_deaths_recoveries, from_df_districtwise])
    
    # set indices to date to combine on it
    from_df_raw_data = from_df_raw_data.set_index(datecol)
    from_df_deaths_recoveries = from_df_deaths_recoveries.set_index(datecol)
    from_df_districtwise = from_df_districtwise.set_index(datecol)
    
    # df_raw_data doesn't contain deceased/recovered numbers, so add them
    out = copy.copy(from_df_raw_data)
    out.loc[from_df_deaths_recoveries.index,from_df_deaths_recoveries.columns] = from_df_deaths_recoveries
    # use df_districtwise data starting from when it starts
    for idx in from_df_districtwise.index:
        if idx not in out.index:
            out = out.append(pd.Series(name=idx))
    out.loc[from_df_districtwise.index,from_df_districtwise.columns] = from_df_districtwise
    # only as up to date as df_districtwise
    out = out.loc[:np.max(from_df_districtwise.index)]
    # warn if more than 2 days out of date. TODO use an actual logger
    if datetime.datetime.today() - np.max(from_df_districtwise.index) > datetime.timedelta(days=2):
        print(f'WARNING: df_districtwise has not been updated in 2 days. Last Updated: {np.max(from_df_districtwise.index)}')
    
    out[districtcol] = district_val
    out[statecol] = state_val
    out.index.name = datecol
    return out.reset_index()

def get_concat_data(dataframes, state, district, new_district_name=None, concat=False):
    if concat:
        if type(district) == list:
            assert new_district_name != None, "must provide new_district_name"
            all_dfs = defaultdict(list)
            for dist in district:
                raw = get_data(dataframes, state=state, district=dist, use_dataframe='raw_data')
                if len(raw) != 0:
                    all_dfs['from_df_raw_data'].append(raw)
                dr = get_data(dataframes, state=state, district=dist, use_dataframe='deaths_recovs')
                if len(dr) != 0:
                    all_dfs['from_df_deaths_recoveries'].append(dr)
                dwise = get_data(dataframes, state=state, district=dist, use_dataframe='districts_daily')
                if len(dwise) != 0:
                    all_dfs['from_df_districtwise'].append(dwise)
            from_df_raw_data = combine_districts(all_dfs['from_df_raw_data'], new_district=new_district_name)
            from_df_deaths_recoveries = combine_districts(all_dfs['from_df_deaths_recoveries'], new_district=new_district_name)
            from_df_districtwise = combine_districts(all_dfs['from_df_districtwise'], new_district=new_district_name)
        else:
            from_df_raw_data = get_data(dataframes, state=state, district=district, use_dataframe='raw_data')
            from_df_deaths_recoveries = get_data(dataframes, state=state, district=district, use_dataframe='deaths_recovs')
            from_df_districtwise = get_data(dataframes, state=state, district=district, use_dataframe='districts_daily')
        return concat_sources(from_df_raw_data, from_df_deaths_recoveries, from_df_districtwise)
    else:
        if type(district) == list:
            assert new_district_name != None, "must provide new_district_name"
            all_dfs = []
            for dist in district:
                dwise = get_data(dataframes, state=state, district=dist, use_dataframe='districts_daily')
                if len(dwise) != 0:
                    all_dfs.append(dwise)
            from_df_districtwise = combine_districts(all_dfs, new_district=new_district_name)
        else:
            from_df_districtwise = get_data(dataframes, state=state, district=district, use_dataframe='districts_daily')
        return from_df_districtwise
