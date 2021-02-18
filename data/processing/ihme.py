import sys
import pandas as pd

sys.path.append('../..')
from data.dataloader import JHULoader, Covid19IndiaLoader

def india_all():
    dlobj = Covid19IndiaLoader()
    dataframes = dlobj.pull_dataframes_cached()['df_india_time_series']
    df = dataframes['df_india_time_series']
    # df.dtypes
    df = df[['date', 'totalconfirmed', 'totaldeceased','totalrecovered']]
    df.columns = [df.columns[0]] + [col[5:] for col in df.columns[1:]]
    return df

def india_all_state():
    dlobj = Covid19IndiaLoader()
    dataframes = dlobj.pull_dataframes_cached()['df_india_time_series']
    df = dataframes['df_districts']
    df = df.groupby(['state', 'date']).sum()
    df.reset_index()
    return df

def jhu(country):
    df_master = JHULoader().get_jhu_data()
    df = df_master[df_master['Country/Region'] == country]
    df.loc[:, 'day'] = (df['Date'] - df['Date'].min()).dt.days
    df.loc[:, 'confirmed'] = pd.to_numeric(df['ConfirmedCases'])
    df.loc[:, 'deceased'] = pd.to_numeric(df['Deaths'])
    df.loc[:, 'recovered'] = pd.to_numeric(df['RecoveredCases'])
    df.loc[:, 'active'] = pd.to_numeric(df['ActiveCases'])
    df['Province/State'].fillna(country, inplace=True)
    df.columns = [c.lower() for c in df.columns]
    return df
