import pandas as pd
import numpy as np
import copy

def get_district_time_series(dataframes, state='Karnataka', district='Bengaluru'):
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
