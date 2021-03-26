import sys
sys.path.append('../../')

import pandas as pd
import copy

import data.dataloader as dl

def get_testing_data(state, dist, dataloader='Covid19IndiaLoader'):
    """Function for getting testing data

    Args:
        state (str): Name of State
        dist (str): Name of District
        dataloader (str, optional): Which dataloader to use. Defaults to 'Covid19IndiaLoader'.

    Returns:
        pd.DataFrame: dataframe with testing data
    """
    dlobj = getattr(dl, dataloader)()
    dataframes = dlobj.pull_dataframes_cached()
    if dataloader == 'AthenaLoader':
        df_testing = copy.copy(dataframes['testing_summary'])
        del df_testing['partition_0']
        del df_testing['new']

        df_testing = df_testing.loc[(df_testing['state'] == state.lower()) & \
            df_testing['district'] == dist.lower(), :]
        df_testing.dropna(axis=0, how='any', inplace=True)
        df_testing['date'] = pd.to_datetime(df_testing['date'])
        df_testing = df_testing.infer_objects()
        df_testing['positives'] = df_testing['positives'].astype('int64')
        df_testing['positives'] = df_testing['positives'].shift(-1)
        df_testing['tests'] = df_testing['tests'].astype('int64')
        df_testing['tpr'] = (df_testing['positives']*100/df_testing['tests'])
        df_testing.dropna(axis=0, how='any', inplace=True)
        df_testing.reset_index(inplace=True)

    if dataloader == 'Covid19IndiaLoader':
        df_testing = dlobj.get_data(state, dist, use_dataframe='data_all')

    return df_testing
