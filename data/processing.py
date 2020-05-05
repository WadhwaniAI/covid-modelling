import pandas as pd
import numpy as np
import copy
import datetime

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
        return df_district
