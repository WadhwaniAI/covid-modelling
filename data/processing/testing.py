import sys
sys.path.append('../../')

import pandas as pd
import copy

from data.dataloader import AthenaLoader
from data.processing.processing import get_dataframes_cached

def get_testing_data(state='Maharashtra', dist='Mumbai'):
    dataframes = get_dataframes_cached(loader_class=AthenaLoader)

    df_testing = copy.copy(dataframes['testing_summary'])
    del df_testing['partition_0']
    del df_testing['new']

    df_testing = df_testing.loc[df_testing['district'] == dist.lower(), :]

    df_testing.dropna(axis=0, how='any', inplace=True)
    df_testing['date'] = pd.to_datetime(df_testing['date'])
    df_testing = df_testing.infer_objects()
    df_testing['positives'] = df_testing['positives'].astype('int64')
    df_testing['positives'] = df_testing['positives'].shift(-1)
    df_testing['tests'] = df_testing['tests'].astype('int64')
    df_testing['tpr'] = (df_testing['positives']*100/df_testing['tests'])
    df_testing.dropna(axis=0, how='any', inplace=True)
    df_testing.reset_index(inplace=True)

    return df_testing
