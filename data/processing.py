import pandas as pd
import numpy as np
import copy
import datetime

from data.dataloader import get_rootnet_api_data

def get_data(dataframes, state, district, use_dataframe='districts_daily', disable_tracker=False, filename=None):
    if disable_tracker:
        df_result = pd.read_csv(filename)
        df_result['date'] = pd.to_datetime(df_result['date'])
        df_result.columns = [x if x != 'active' else 'hospitalised' for x in df_result.columns]
        df_result.columns = [x if x != 'confirmed' else 'total_infected' for x in df_result.columns]
        #TODO add support of adding 0s column for the ones which don't exist
        return df_result
    if district != None:
        df_result = get_district_time_series(
            dataframes, state=state, district=district, use_dataframe=use_dataframe)
    else:
        df_result = get_state_time_series(state=state)
    return df_result

def get_state_time_series(state='Delhi'):
    rootnet_dataframes = get_rootnet_api_data()
    df_states = rootnet_dataframes['df_state_time_series']
    df_state = df_states[df_states['state'] == state]
    df_state = df_state.loc[df_state['date'] >= '2020-04-24', :]
    df_state = df_state.loc[df_state['date'] < datetime.date.today().strftime("%Y-%m-%d"), :]
    df_state.reset_index(inplace=True, drop=True)
    return df_state

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
        return df_district
