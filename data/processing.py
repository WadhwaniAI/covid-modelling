import pandas as pd
import numpy as np
import copy
import datetime

from data.dataloader import get_rootnet_api_data

def get_data(dataframes=None, state=None, district=None, use_dataframe='districts_daily', disable_tracker=False,
             filename=None, data_format='new'):
    if disable_tracker:
        df_result = get_custom_data(filename, data_format=data_format)
    elif district != None:
        df_result = get_district_time_series(dataframes, state=state, district=district, use_dataframe=use_dataframe)
    else:
        df_result = get_state_time_series(state=state)
    return df_result
    
#TODO add support of adding 0s column for the ones which don't exist
def get_custom_data(filename, data_format='new'):
    if data_format == 'new':
        df = pd.read_csv(filename)
        del df['Ward/block name']
        del df['Ward number (if applicable)']
        del df['Mild cases (isolated)']
        del df['Moderate cases (hospitalized)']
        del df['Severe cases (In ICU)']
        del df['Critical cases (ventilated patients)']
        df.columns = ['state', 'district', 'date', 'total_infected', 'hospitalised', 'recovered', 'deceased']
        df.drop(np.arange(3), inplace=True)
        df['date'] = pd.to_datetime(df['date'], format='%m-%d-%Y')
        df = df[np.logical_not(df['state'].isna())]
        df.reset_index(inplace=True, drop=True)
        df.loc[:, ['total_infected', 'hospitalised', 'recovered', 'deceased']] = df[[
            'total_infected', 'hospitalised', 'recovered', 'deceased']].apply(pd.to_numeric)
        df = df[['date', 'state', 'district', 'total_infected', 'hospitalised', 'recovered', 'deceased']]
        return df
    if data_format == 'old':
        df_result = pd.read_csv(filename)
        df_result['date'] = pd.to_datetime(df_result['date'])
        df_result.columns = [x if x != 'active' else 'hospitalised' for x in df_result.columns]
        df_result.columns = [x if x != 'confirmed' else 'total_infected' for x in df_result.columns]
        

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
