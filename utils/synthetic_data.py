import pandas as pd
import yaml

from datetime import timedelta

from utils.enums import Columns
from utils.util import train_test_split


def create_custom_dataset(df_actual, df_synthetic, use_actual=True, use_synthetic=True,
                          start_date=None, allowance=5, s1=15, s2=15, s3=15):
    """Creates a custom dataset from ground truth and synthetic data
    Considers the dataset to consist of three series - s1, s2, s3
    s1 + s2 is expected to form the train split and has 3 possibilities:
        s1 and s2 are both ground truth data
        s1 is ground truth data and s2 is synthetic data
        s1 and s2 are synthetic data
    s3 is expected to form the test split and is created from ground truth data only

    Args:
        df_actual (pd.DataFrame): ground truth data
        df_synthetic (pd.DataFrame): synthetic data
        use_actual (bool, optional): whether to use ground truth data in custom dataset (default: True)
        use_synthetic (bool, optional): whether to use synthetic data in custom dataset (default: True)
        start_date (str, optional): date from which custom dataset should start (default: None)
        allowance (int, optional): number of days of ground data before train split to prevent nan when rolling average
            is taken (default: 5)
        s1 (int, optional): length of series 1 (default: 15)
        s2 (int, optional): length of series 2 (default: 15)
        s3 (int, optional): length of series 3 (default: 15)

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame, dict: custom dataset, train split, test split, dataset properties
    """

    # Set start date of dataset (including allowance for rolling average)
    if start_date is None:
        start_date = df_actual['date'].min()
    else:
        start_date = pd.to_datetime(start_date, dayfirst=False)

    # Remove data before the start date
    threshold = start_date - timedelta(days=1)

    _, df_actual = train_test_split(df_actual, threshold)
    if df_synthetic is not None:
        _, df_synthetic = train_test_split(df_synthetic, threshold)

    test_size = s3
    end_of_train = start_date + timedelta(allowance + s1 + s2 - 1)

    # Create train and test splits using ground truth and synthetic data
    df_train, df_test = train_test_split(df_actual, end_of_train)
    if not use_synthetic:  # Use only ground truth data
        pass
    else:
        if df_synthetic is None:
            raise Exception("No synthetic dataset provided.")
        elif not use_actual:  # Use only synthetic data
            df_train, _ = train_test_split(df_synthetic, end_of_train)
        elif use_actual and use_synthetic:  # s1 from ground truth and s2 from synthetic
            end_of_actual = start_date + timedelta(s1 + allowance - 1)
            df_train1, _ = train_test_split(df_actual, end_of_actual)
            df_train_temp, _ = train_test_split(df_synthetic, end_of_train)
            _, df_train2 = train_test_split(df_train_temp, end_of_actual)
            df_train = pd.concat([df_train1, df_train2], axis=0)
        else:
            raise Exception("Train and test sets not defined.")
    df_train.reset_index(inplace=True, drop=True)

    # Remove data after s3
    if len(df_test) > test_size:
        df_test = df_test.head(test_size)
    else:
        raise Exception("Test set size {} greater than size available {}.".format(test_size, len(df_test)))

    df = pd.concat([df_train, df_test], axis=0)

    # Create dict of dataset properties - remove?
    properties = {
        "allowance_before_train": allowance,
        "total_length_of_dataset": allowance + s1 + s2 + s3,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "train_length (s1+s2)": s1 + s2,
        "test_length (s3)": test_size,
        "use_synthetic_s1": (not use_actual),
        "use_synthetic_s2": (use_actual and use_synthetic)
    }

    return df, df_train, df_test, properties


def format_custom_dataset(dataset, state=None, district=None, compartments=None):
    """Formats the custom dataset according to the format required by the SEIR model
    Inserts the required columns and removes others

    Args:
        dataset (pd.DataFrame): custom dataset
        state (str, optional): name of state
            (default: {None} on the assumption that the dataset already contains this column)
        district (str, optional): name of district
            (default: {None} on the assumption that the dataset already contains this column)
        compartments (list, optional): list of compartments to include (default: None)

    Returns:
        pd.DataFrame: custom dataset with required columns
    """

    # Set the columns of compartments to be included
    if compartments is None:
        col_names = [col.name for col in Columns.curve_fit_compartments()]
    else:
        col_names = compartments

    # Include or exclude required columns
    dataset = dataset[['date']+col_names]
    if state:
        dataset.insert(1, "state", state)
    if district:
        dataset.insert(2, "district", district)
    return dataset


def get_experiment_dataset(district, state, original_data, generated_data=None, use_actual=True, use_synthetic=True,
                           start_date=None, allowance=5, s1=15, s2=15, s3=15):
    """Returns formatted custom dataset for an experiment

    Args:
        district (str): name of district
        state (str): name of state
        original_data (pd.DataFrame): ground truth data
        generated_data (pd.DataFrame, optional): synthetic data
        use_actual (bool, optional): whether to use ground truth data in custom dataset (default: True)
        use_synthetic (bool, optional): whether to use synthetic data in custom dataset (default: True)
        start_date (str, optional): date from which custom dataset should start including allowance (default: None)
        allowance (int, optional): number of days of ground data before train split to prevent nan when rolling average
            is taken (default: 5)
        s1 (int, optional): length of series 1 (default: 15)
        s2 (int, optional): length of series 2 (default: 15)
        s3 (int, optional): length of series 3 (default: 15)

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame, dict:
            formatted custom dataset, train split, test split, dataset properties
    """
    df, train, test, prop = create_custom_dataset(original_data, generated_data,
                                                  use_actual=use_actual, use_synthetic=use_synthetic,
                                                  start_date=start_date, allowance=allowance, s1=s1, s2=s2, s3=s3)
    df = format_custom_dataset(df, state, district)
    train = format_custom_dataset(train, state, district)
    test = format_custom_dataset(test, state, district)
    return df, train, test, prop


def insert_custom_dataset_into_dataframes(dataframes, dataset, start_date=None, compartments=None):
    """Replaces original df_district of SEIR input dataframe with one or more columns from custom dataset

    Args:
        dataframes (tuple(pd.DataFrame, pd.DataFrame)): data from main source and data from raw_data in covid19india
        dataset (pd.Dataframe): custom dataset
        start_date (str, optional): date from which dataset should start (default: None)
        compartments (list, optional): compartments for which ground truth data is replaced by synthetic data

    Returns:
        pd.DataFrame, pd.DataFrame: data from main source with synthetic data added and data from covid19india raw_data
    """

    # Set the columns of compartments to be replaced
    if compartments is None:
        col_names = [col.name for col in Columns.curve_fit_compartments()]
    else:
        col_names = compartments

    df_district, df_raw = dataframes

    # Set start date of dataset (including allowance for rolling average)
    if start_date is None:
        start_date = df_district['date'].min()
    else:
        start_date = pd.to_datetime(start_date, dayfirst=False)

    # Remove data before the start date
    threshold = start_date - timedelta(days=1)

    _, df_district = train_test_split(df_district, threshold)
    _, df_raw = train_test_split(df_raw, threshold)

    # Replace compartment columns in df_district with synthetic data and clip to have as many rows as the custom dataset
    num_rows = dataset.shape[0]
    df_district = df_district.head(num_rows)
    for col in col_names:
        df_district[col] = dataset[col].values

    return df_district, df_raw


def read_synth_data_config(path):
    """Reads config file for synthetic data generation experiments

    Args:
        path (str): path to config file

    Returns:
        dict: config for synthetic data generation experiments
    """

    with open(path) as configfile:
        config = yaml.load(configfile, Loader=yaml.SafeLoader)
    config = config['base']
    return config
