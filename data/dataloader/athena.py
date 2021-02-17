from pyathena import connect
import copy
import pandas as pd
import numpy as np
from pyathena.pandas_cursor import PandasCursor

from data.dataloader.base import BaseLoader


class AthenaLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def create_connection(self, schema, staging_dir, pyathena_rc_path=None):
        """Creates SQL Server connection using AWS Athena credentials

        Keyword Arguments:
            pyathena_rc_path {str} -- [Path to the PyAthena RC file with the AWS Athena variables] (default: {None})

        Returns:
            [cursor] -- [Connection Cursor]
        """
        if pyathena_rc_path is None:
            pyathena_rc_path = '../../misc/pyathena/pyathena.rc'
        SCHEMA_NAME = schema

        # Open Pyathena RC file and get list of all connection variables in a processable format
        with open(pyathena_rc_path) as f:
            lines = f.readlines()

        lines = [x.strip() for x in lines]
        lines = [x.split('export ')[1] for x in lines]
        lines = [line.replace('=', '="') + '"' if '="' not in line else line for line in lines]
        lines = [line.replace('S3_STAGING_DIR_NAME', staging_dir) for line in lines]
        variables = [line.split('=') for line in lines]

        # Create variables using the processed variable names from the RC file
        AWS_CREDS = {}
        for key, var in variables:
            exec("{} = {}".format(key, var), AWS_CREDS)

        # Create connection
        cursor = connect(aws_access_key_id=AWS_CREDS['AWS_ACCESS_KEY_ID'],
                        aws_secret_access_key=AWS_CREDS['AWS_SECRET_ACCESS_KEY'],
                        s3_staging_dir=AWS_CREDS['AWS_ATHENA_S3_STAGING_DIR'],
                        region_name=AWS_CREDS['AWS_DEFAULT_REGION'],
                        work_group=AWS_CREDS['AWS_ATHENA_WORK_GROUP'],
                        schema_name=SCHEMA_NAME).cursor(PandasCursor)
        return cursor


    def pull_dataframes(self, schema, tables, staging_dir, pyathena_rc_path=None, **kwargs):
        """Creates connection to Athena database and returns all the tables there as a dict of Pandas dataframes

        Keyword Arguments:
            pyathena_rc_path {str} -- Path to the PyAthena RC file with the AWS Athena variables 
            (If you don't have this contact jerome@wadhwaniai.org) (default: {None})

        Returns:
            dict -- dict where key is str and value is pd.DataFrame
            The dataframes : 
            covid_case_summary
            new_covid_case_summary
            demographics_details
            healthcare_capacity
            testing_summary
        """
        if pyathena_rc_path is None:
            pyathena_rc_path = '../../misc/pyathena/pyathena.rc'

        # Create connection
        cursor = self.create_connection(schema, staging_dir, pyathena_rc_path)

        # Run SQL SELECT queries to get all the tables in the database as pandas dataframes
        dataframes = {}
        for table in tables:
            dataframes[table] = cursor.execute(
                'SELECT * FROM {}'.format(table)).as_pandas()

        return dataframes

    def pull_dataframes_cached(self, reload_data=False, label=None, **kwargs):
        return super().pull_dataframes_cached(reload_data=reload_data, label=label, **kwargs)

    def get_data(self, state='Maharashtra', district='Mumbai', reload_data=False, **kwargs):
        print('fetching from athenadb...')
        label = kwargs.pop('label', None)
        dataframes = self.pull_dataframes_cached(reload_data=reload_data, **kwargs)
        df_result = copy.copy(dataframes['case_summaries'])
        df_result.rename(columns={'deaths': 'deceased', 'total cases': 'total',
                                'active cases': 'active', 'recoveries': 'recovered'}, inplace=True)
        df_result = df_result[np.logical_and(
            df_result['state'] == state, df_result['district'] == district)]
        df_result = df_result.loc[:, :'deceased']
        df_result.dropna(axis=0, how='any', inplace=True)
        df_result.loc[:, 'date'] = pd.to_datetime(df_result['date'])
        df_result.reset_index(inplace=True, drop=True)
        for col in df_result.columns:
            if col in ['active', 'total', 'recovered', 'deceased']:
                df_result[col] = df_result[col].astype('int64')

        return {"data_frame": df_result}
