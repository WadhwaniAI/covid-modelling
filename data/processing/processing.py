import copy
import datetime

import data.dataloader as dl


def get_data(dataloader, dataloading_params, data_columns):
    """Main function that instantiates the dataloader, and gets the data from it 
    according to the dataloading_params

    Args:
        dataloader (str): The name of the dataloader class
        dataloading_params (dict): The dict of dataloading params
        data_columns (list[str]): The list of columns the returned dataframe is expected to return

    Raises:
        ValueError: Raises value error if the returned dataframe does not contain the columns in `data_columns`

    Returns:
        pd.DataFrame: Processed dataframe
    """
    dl_class = getattr(dl, dataloader)
    dlobj = dl_class()
    res = dlobj.get_data(**dataloading_params)
    if set(data_columns).issubset(set(res['data_frame'].columns)):
        return res
    else:
        raise ValueError('The returned dataframe from the specified dataloader (`dataloader`) doesn\'t' +
            'contain the specified columns (`data_columns`)')


def implement_rolling(df, window_size, center, win_type, min_periods):
    """Helper function for implementing rolling average for ALL numeric columns

    Args:
        df (pd.DataFrame): The dataframe to implement rolling on
        window_size (int): Window size 
        center (bool): If true, a centered window is used
        win_type (str): Which window type
        min_periods (int): Towards the ends of the dataframe, if the `window_size` 
        becomes smaller than `min_periods`, the ground truth value is used 
        instead of the rolling average one.

    Returns:
        pd.DataFrame: Dataframe with rolling averaged values
    """
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
    """Helper function for implementing train val test split

    Args:
        df (pd.DataFrame): df to implement split on
        train_period (int): Train period
        val_period (int): Val period
        test_period (int): Test period
        start_date (datetime.date/int): Starting date. If int, df.iloc[start_date, 'date'] 
        is assumed as start_date
        end_date (datetime.date/int): Ending date. If int, df.iloc[end_date, 'date'] 
        is assumed as end_date

    Raises:
        ValueError: Raises error if both start and end date are None
        ValueError: Raises error if start date is int and < 0
        ValueError: Raises error if end date is int and > 0

    Returns:
        pd.DataFrame*3: Train val test dfs
    """
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
    """Function for implementing rolling average AND train val test split

    Args:
        df_district (pd.DataFrame): df to do split + rolling avg on
        train_period (int, optional): Train period. Defaults to 5.
        val_period (int, optional): Val period. Defaults to 5.
        test_period (int, optional): Test period. Defaults to 5.
        start_date (datetime.date/int, optional): Starting Date. Defaults to None.
        end_date (datetime.date/int, optional): Ending Date. Defaults to None.
        window_size (int, optional): Window Size. Defaults to 5.
        center (bool, optional): If true, a centered window is used. Defaults to True.
        win_type (str, optional): Special pandas window type. Defaults to None.
        min_periods (int): Towards the ends of the dataframe, if the `window_size` 
        becomes smaller than `min_periods`, the ground truth value is used. Defaults to 1.
        split_after_rolling (bool, optional): If true, splitting is done after rolling average. Defaults to False.

    Returns:
        pd.DataFrame*3: dataframe with train val test split (processed with rolling avg)
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
