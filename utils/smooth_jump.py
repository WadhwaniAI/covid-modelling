import numpy as np
import pandas as pd
import datetime

def smooth_big_jump(df_district, smoothing_length, data_from_tracker, t_recov=14, method='uniform'):
    if data_from_tracker:
        d1, d2 = '2020-05-29', '2020-05-30'
    else:
        d1, d2 = '2020-05-28', '2020-05-29'
    df_district['date'] = pd.to_datetime(df_district['date'])
    df_district = df_district.set_index('date')
    big_jump = df_district.loc[d2, 'recovered'] - df_district.loc[d1, 'recovered']
    print(big_jump)
    if method == 'uniform':
        for i, day_number in enumerate(range(smoothing_length-2, -1, -1)):
            date = datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.timedelta(days=day_number)
            offset = np.random.binomial(1, (big_jump%smoothing_length)/smoothing_length)
            df_district.loc[date, 'recovered'] += ((i+1)*big_jump)//smoothing_length + offset
            df_district.loc[date, 'hospitalised'] -= ((i+1)*big_jump)//smoothing_length + offset

    elif method == 'weighted':
        newcases = df_district['total_infected'].shift(t_recov) - df_district['total_infected'].shift(t_recov + 1)
        valid_idx = newcases.first_valid_index()
        window_start = datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.timedelta(days=smoothing_length - 1)
        newcases = newcases.loc[max(valid_idx, window_start):d1]
        truncated = df_district.loc[max(valid_idx, window_start):d1, :]
        smoothing_length = len(truncated)
        print(f'smoothing length truncated to {smoothing_length}')
        invpercent = newcases.sum()/newcases
        for day_number in range(smoothing_length-1, -1, -1):
            date = datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.timedelta(days=day_number)
            offset = np.random.binomial(1, (big_jump%invpercent.loc[date])/invpercent.loc[date])
            truncated.loc[date:, 'recovered'] += (big_jump // invpercent.loc[date]) + offset
            truncated.loc[date:, 'hospitalised'] -= (big_jump // invpercent.loc[date]) + offset
        df_district.loc[truncated.index, 'recovered'] = truncated['recovered'].astype('int64')
        df_district.loc[truncated.index, 'hospitalised'] = truncated['hospitalised'].astype('int64')

    elif method == 'weighted-recov':
        newcases = df_district['recovered'].shift(0) - df_district['recovered'].shift(1)
        valid_idx = newcases.first_valid_index()
        window_start = datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.timedelta(days=smoothing_length - 1)
        newcases = newcases.loc[max(valid_idx, window_start):d1]
        truncated = df_district.loc[max(valid_idx, window_start):d1, :]
        smoothing_length = len(truncated)
        print(f'smoothing length truncated to {smoothing_length}')
        invpercent = newcases.sum()/newcases
        for day_number in range(smoothing_length-1, -1, -1):
            date = datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.timedelta(days=day_number)
            offset = np.random.binomial(1, (big_jump%invpercent.loc[date])/invpercent.loc[date])
            truncated.loc[date:, 'recovered'] += (big_jump // invpercent.loc[date]) + offset
            truncated.loc[date:, 'hospitalised'] -= (big_jump // invpercent.loc[date]) + offset
        df_district.loc[truncated.index, 'recovered'] = truncated['recovered'].astype('int64')
        df_district.loc[truncated.index, 'hospitalised'] = truncated['hospitalised'].astype('int64')

    else:
        raise Exception("unknown smoothing method provided")
    
    assert((df_district['total_infected'] == df_district['hospitalised'] + df_district['deceased'] + df_district['recovered']).all())
    return df_district.reset_index()
