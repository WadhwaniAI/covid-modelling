import pandas as pd
import numpy as np


def get_district_time_series(dataframes, state='Karnataka', district='Bengaluru'):
    df_raw_data_1 = dataframes['df_raw_data'][dataframes['df_raw_data']['detectedstate'] == state]
    df_raw_data_1 = df_raw_data_1[df_raw_data_1['detecteddistrict'] == district]
    df_raw_data_1['dateannounced'] = pd.to_datetime(df_raw_data_1['dateannounced'], format='%d/%m/%Y')

    index = pd.date_range(np.min(df_raw_data_1['dateannounced']), np.max(df_raw_data_1['dateannounced']))

    df_district = pd.DataFrame(columns=['total_confirmed'], index=index)
    df_district['total_confirmed'] = [0]*len(index)
    for _, row in df_raw_data_1.iterrows():
        df_district.loc[row['dateannounced']:, 'total_confirmed'] += 1

    df_district.reset_index(inplace=True)
    df_district.columns = ['date', 'total_confirmed']
    return df_district
