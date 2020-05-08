import pandas as pd
import numpy as np
import copy
import datetime
from collections import defaultdict

def get_data(dataframes, state, district, use_dataframe='districts_daily', disable_tracker=False, filename=None):
    if disable_tracker:
        df_district = pd.read_csv(filename)
        df_district['date'] = pd.to_datetime(df_district['date'])
        #TODO add support of adding 0s column for the ones which don't exist
        return df_district

    df_district = get_district_time_series(
        dataframes, state=state, district=district, use_dataframe=use_dataframe)
    return df_district

def get_district_time_series(dataframes, state='Karnataka', district='Bengaluru', use_dataframe='raw_data'):
    if use_dataframe == 'districts_daily':
        df_districts = copy.copy(dataframes['df_districts'])
        df_district = df_districts[np.logical_and(df_districts['state'] == state, df_districts['district'] == district)]
        del df_district['notes']
        df_district.loc[:, 'date'] = pd.to_datetime(df_district.loc[:, 'date'])
        df_district = df_district.loc[df_district['date'] >= '2020-04-24', :]
        df_district = df_district.loc[df_district['date'] < datetime.date.today().strftime("%Y-%m-%d"), :]
        df_district.columns = [x if x != 'active' else 'hospitalised' for x in df_district.columns]
        df_district.columns = [x if x != 'confirmed' else 'total_infected' for x in df_district.columns]
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
                'total_infected':pd.Series([], dtype='int'), 
                'hospitalised':pd.Series([], dtype='int'), 
                'deceased':pd.Series([], dtype='int'), 
                'recovered':pd.Series([], dtype='int'),
                'district':pd.Series([], dtype='object'),
                'state':pd.Series([], dtype='object'),
                })
            out.index.name = 'date'
            return out.reset_index()
        
        df_raw_data_1['dateannounced'] = pd.to_datetime(df_raw_data_1['dateannounced'], format='%d/%m/%Y')

        index = pd.date_range(np.min(df_raw_data_1['dateannounced']), np.max(df_raw_data_1['dateannounced']))

        df_district = pd.DataFrame(columns=['total_infected'], index=index)
        df_district['total_infected'] = [0]*len(index)
        for _, row in df_raw_data_1.iterrows():
            try:
                df_district.loc[row['dateannounced']:, 'total_infected'] += 1*int(row['numcases'])
            except Exception:
                df_district.loc[row['dateannounced']:, 'total_infected'] += 1

        df_district.reset_index(inplace=True)
        df_district.columns = ['date', 'total_infected']
        df_district['hospitalised'] = [0]*len(df_district)
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

def get_concat_data(dataframes, state, district, new_district_name=None):
    if type(district) == list:
        assert new_district_name != None, "must provide new_district_name"
        all_dfs = defaultdict(list)
        for dist in district:
            raw = get_district_time_series(dataframes, state, dist, use_dataframe='raw_data')
            if len(raw) != 0:
                all_dfs['from_df_raw_data'].append(raw)
            dr = get_district_time_series(dataframes, state, dist, use_dataframe='deaths_recovs')
            if len(dr) != 0:
                all_dfs['from_df_deaths_recoveries'].append(dr)
            dwise = get_district_time_series(dataframes, state, dist, use_dataframe='districts_daily')
            if len(dwise) != 0:
                all_dfs['from_df_districtwise'].append(dwise)
        from_df_raw_data = combine_districts(all_dfs['from_df_raw_data'], new_district=new_district_name)
        from_df_deaths_recoveries = combine_districts(all_dfs['from_df_deaths_recoveries'], new_district=new_district_name)
        from_df_districtwise = combine_districts(all_dfs['from_df_districtwise'], new_district=new_district_name)
    else:
        from_df_raw_data = get_district_time_series(dataframes, state, district, use_dataframe='raw_data')
        from_df_deaths_recoveries = get_district_time_series(dataframes, state, district, use_dataframe='deaths_recovs')
        from_df_districtwise = get_district_time_series(dataframes, state, district, use_dataframe='districts_daily')
    return concat_sources(from_df_raw_data, from_df_deaths_recoveries, from_df_districtwise)

