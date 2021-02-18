import copy
import datetime

import data.dataloader as dl


def get_data(data_source, dataloading_params):
    """

    """
    try:
        dl_class = getattr(dl, data_source)
        dlobj = dl_class()
        return dlobj.get_data(**dataloading_params)
    except Exception as e:
        print(e)


def implement_rolling(df, window_size, center, win_type, min_periods):
    df_roll = df.infer_objects()
    # Select numeric columns
    which_columns = df_roll.select_dtypes(include='number').columns
    for column in which_columns:
        df_roll[column] = df_roll[column].rolling(window=window_size, center=center, win_type=win_type, 
                                                  min_periods=min_periods).mean()
        # For the days which become na after rolling, the following line 
        # uses the true observations inplace of na, and the rolling average where it exists
        df_roll[column] = df_roll[column].fillna(df[column])

    return df_roll

def implement_split(df, train_period, val_period, test_period, start_date, end_date):
    if start_date is not None and end_date is not None:
        raise ValueError('Both start_date and end_date cannot be specified. Please specify only 1')
    elif start_date is not None:
        if isinstance(start_date, int):
            if start_date < 0:
                raise ValueError('Please enter a positive value for start_date if entering an integer')
        if isinstance(start_date, datetime.date):
            start_date = df.loc[df['date'].dt.date == start_date].index[0]

        df_train = df.iloc[:start_date + train_period, :]
        df_val = df.iloc[start_date + train_period:start_date + train_period + val_period, :]
        df_test = df.iloc[start_date + train_period + val_period: \
                          start_date + train_period + val_period + test_period, :]
    else:    
        if end_date is not None:
            if isinstance(end_date, int):
                if end_date > 0:
                    raise ValueError('Please enter a negative value for end_date if entering an integer')
            if isinstance(end_date, datetime.date):
                end_date = df.loc[df['date'].dt.date == end_date].index[0] - len(df) + 1
        else:
            end_date = 0  

        df_test = df.iloc[len(df) - test_period+end_date:end_date, :]
        df_val = df.iloc[len(df) - (val_period+test_period) +
                        end_date:len(df) - test_period+end_date, :]
        df_train = df.iloc[:len(df) - (val_period+test_period)+end_date, :]

    return df_train, df_val, df_test

def train_val_test_split(df_district, train_period=5, val_period=5, test_period=5, start_date=None, end_date=None,  
                         window_size=5, center=True, win_type=None, min_periods=1, split_after_rolling=False):
    """Creates train val split on dataframe

    # TODO : Add support for creating train val test split

    Arguments:
        df_district {pd.DataFrame} -- The observed dataframe

    Keyword Arguments:
        val_period {int} -- Size of val set (default: {5})
        window_size {int} -- Size of rolling window. The rolling window is centered (default: {5})

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame -- train dataset, val dataset, concatenation of rolling average dfs
    """
    print("splitting data ..")
    df_district_rolling = copy.copy(df_district)
    # Perform rolling average on all columns with numeric datatype
    if split_after_rolling:
        df_district_rolling = implement_rolling(
            df_district_rolling, window_size, center, win_type, min_periods)
        df_train, df_val, df_test = implement_split(df_district_rolling, train_period, val_period, 
                                                    test_period, start_date, end_date)
        
    else:
        df_train, df_val, df_test = implement_split(df_district_rolling, train_period, val_period,
                                                    test_period, start_date, end_date)

        df_train = implement_rolling(df_train, window_size, center, win_type, min_periods)
        df_val = implement_rolling(df_val, window_size, center, win_type, min_periods)
        df_test = implement_rolling(df_test, window_size, center, win_type, min_periods)

    df_train = df_train.infer_objects()
    df_val = df_val.infer_objects()
    df_test = df_test.infer_objects()
        
    if val_period == 0:
        df_val = None

    return df_train, df_val, df_test
