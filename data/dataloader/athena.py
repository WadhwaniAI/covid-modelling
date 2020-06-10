import pandas as pd
import numpy as np
import datetime

from pyathena import connect
from pyathena.pandas_cursor import PandasCursor

from data.dataloader.base import BaseLoader


class AthenaLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def create_connection(self, pyathena_rc_path=None):
        """Creates SQL Server connection using AWS Athena credentials

        Keyword Arguments:
            pyathena_rc_path {str} -- [Path to the PyAthena RC file with the AWS Athena variables] (default: {None})

        Returns:
            [cursor] -- [Connection Cursor]
        """
        if pyathena_rc_path == None:
            pyathena_rc_path = '../../misc/pyathena/pyathena.rc'
        SCHEMA_NAME = 'wiai-covid-data'

        # Open Pyathena RC file and get list of all connection variables in a processable format
        with open(pyathena_rc_path) as f:
            lines = f.readlines()

        # import pdb; pdb.set_trace()
        lines = [x.strip() for x in lines]
        lines = [x.split('export ')[1] for x in lines]
        lines = [line.replace('=', '="') + '"' if '="' not in line else line for line in lines]
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


    def load_data(self, pyathena_rc_path=None):
        """Creates connection to Athena database and returns all the tables there as a dict of Pandas dataframes

        Keyword Arguments:
            pyathena_rc_path {str} -- Path to the PyAthena RC file with the AWS Athena variables 
            (If you don't have this contact jerome@wadhwaniai.org) (default: {None})

        Returns:
            dict -- dict where key is str and value is pd.DataFrame
            The dataframes : 
            covid_case_summary
            demographics_details
            healthcare_capacity
            testing_summary
        """
        if pyathena_rc_path == None:
            pyathena_rc_path = '../../misc/pyathena/pyathena.rc'

        # Create connection
        cursor = self.create_connection(pyathena_rc_path)

        # Run SQL SELECT queries to get all the tables in the database as pandas dataframes
        dataframes = {}
        tables_list = cursor.execute('Show tables').as_pandas().to_numpy().reshape(-1, )
        for table in tables_list:
            dataframes[table] = cursor.execute(
                'SELECT * FROM {}'.format(table)).as_pandas()
        
        return dataframes

    def get_athena_dataframes(self, **kwargs):
        return self.load_data(**kwargs)