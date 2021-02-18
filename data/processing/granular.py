import pandas as pd
import numpy as np
import copy

from data.dataloader import AthenaLoader

"""
The set of functions of processing data with more columns (Active split into multiple columns)
"""


def get_data(data_source, dataloading_params):
    """Handshake between data module and training module. 
        Returns a dataframe of cases from either a filename of AthenaDB

    if filename == None : data loaded from AWS Athena Database
    else : data loded from filename

    Keyword Arguments:
        filename {str} -- Path to CSV file with data (default: {None})

    Returns:
        pd.DataFrame -- dataframe of cases for a area with multiple columns (more than 4)
       
    """
    if data_source == 'filename':
        df_result = get_custom_data_from_file(**dataloading_params)
    elif data_source == 'athena':
        df_result = get_custom_data_from_db(**dataloading_params)
    else:
        raise ValueError('Given input of data_source is incorrect. please either give "athena" or "filename"')
    
    return df_result


def get_custom_data_from_file(filename, state='Maharashtra', district='Mumbai'):
    df = pd.read_csv(filename)
    # Make the first row the column
    df.columns = df.loc[0].apply(lambda x: x.lower().strip().replace(' ', '_'))
    # Delete the first 3 rows (not data)
    df.drop(np.arange(3), inplace=True)
    # Delete all rows where at least 1 datapoint is NaN
    df.dropna(axis=0, how='any', inplace=True)
    # Replace all occurances of ',' in data with ''
    df.replace(',', '', regex=True, inplace=True)
    # Convert to numeric
    df.loc[:, 'total_cases':] = df.loc[:, 'total_cases':].apply(pd.to_numeric)
    # Convert datetime
    df['date'] = pd.to_datetime(df['date'])
    # Infer datetime
    df = df.infer_objects()
    # Delete rows where 1 or more datapoints are < 0
    df = df[(df.select_dtypes(include='int64') > 0).sum(axis=1) == len(df.select_dtypes(include='int64').columns)]
    df.reset_index(inplace=True, drop=True)
    # Column renaming and pruning
    df = df.drop([x for x in df.columns if '_capacity' in x], axis=1)
    df.columns = [x.replace('_occupied', '') for x in df.columns]
    df = df.rename({'total_cases': 'total', 'active_cases': 'active',
                    'icu_beds': 'icu', 'ventilator_beds': 'ventilator',
                    'stable_symptomatic': 'symptomatic', 'stable_asymptomatic': 'asymptomatic', 
                    'recoveries': 'recovered', 'deaths': 'deceased'}, axis='columns')
    # New column creation
    df['hq'] = df['active'] - df['total_beds']
    df['non_o2_beds'] = df['total_beds'] - (df['o2_beds']+df['icu'])

    # Rearranging columns
    col = df.pop('hq')
    df.insert(int(np.where(df.columns == 'o2_beds')[0][0]), 'hq', col)

    col = df.pop('total_beds')
    df.insert(int(np.where(df.columns == 'o2_beds')[0][0]), 'total_beds', col)

    col = df.pop('non_o2_beds')
    df.insert(int(np.where(df.columns == 'o2_beds')[0][0]), 'non_o2_beds', col)

    # Data checks
    beds_check = sum(df.loc[:, ['hq', 'non_o2_beds', 'o2_beds', 'icu']].sum(axis=1) == df['active'])
    facility_check = sum(df.loc[:, ['ccc2', 'dchc', 'dch']].sum(axis=1) == df['total_beds'])
    severity_check = sum(df.loc[:, ['asymptomatic', 'symptomatic', 'critical']].sum(axis=1) == df['active'])
    return df

def get_custom_data_from_db(state='Maharashtra', district='Mumbai', **kwargs):
    print('fetching from athenadb...')
    dlobj = AthenaLoader()
    dataframes = dlobj.pull_dataframes_cached(**kwargs)
    df = copy.copy(dataframes['case_summaries'])
    df_bed = copy.copy(dataframes['bed_summaries'])
    df.dropna(axis=0, how='any', inplace=True)
    df = df.merge(df_bed, on=['state', 'district', 'date', 'partition_0'], how='outer')
    df.rename(columns={'deaths': 'deceased', 'total cases': 'total',
                       'active cases': 'active', 'recoveries': 'recovered'}, inplace=True)
    df.replace(',', '', regex=True, inplace=True)
    df = df[np.logical_and(df['state'] == state, df['district'] == district)]
    df.loc[:, 'total':'ventilator beds occupied'] = df.loc[:, 'total':'ventilator beds occupied']
    df['date'] = pd.to_datetime(df['date'])
    df = df.infer_objects()
    df = df[(df.select_dtypes(include='int64') > 0).sum(axis=1) == len(df.select_dtypes(include='int64').columns)]
    df.reset_index(inplace=True, drop=True)

    # Column renaming and pruning
    df = df.drop([x for x in df.columns if '_capacity' in x] + ['partition_0'], axis=1)
    df = df.rename({'total beds occupied': 'total_beds', 'o2 beds occupied': 'o2_beds', 'icu beds occupied': 'icu',
                    'stable symptomatic': 'symptomatic', 'stable asymptomatic': 'asymptomatic',
                    'ventilator beds occupied': 'ventilator'}, axis='columns')
    # New column creation
    df['hq'] = df['active'] - df['total_beds']
    df['non_o2_beds'] = df['total_beds'] - (df['o2_beds']+df['icu'])

    # Rearranging columns
    col = df.pop('hq')
    df.insert(int(np.where(df.columns == 'o2_beds')[0][0]), 'hq', col)

    col = df.pop('total_beds')
    df.insert(int(np.where(df.columns == 'o2_beds')[0][0]), 'total_beds', col)

    col = df.pop('non_o2_beds')
    df.insert(int(np.where(df.columns == 'o2_beds')[0][0]), 'non_o2_beds', col)
    date_difference = df['date'].diff()
    date_difference.fillna(pd.Timedelta(days=2), inplace=True)
    df = df[df['date'].diff() == pd.Timedelta(days=1)]
    df.reset_index(inplace=True, drop=True)
    return df
