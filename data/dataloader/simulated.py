import pandas as pd
import numpy as np
import os
import yaml
import copy

from data.dataloader.base import BaseLoader
from models.seir import *

class SimulatedDataLoader(BaseLoader):
    """Dataloader that simulates data (ie, generates data from casecounts of a given model, eg SEIR)å

    Args:
        BaseLoader (abstract class): Abstract Data Loader Class
    """
    def __init__(self):
        super().__init__()

    def pull_dataframes(self, **config):
        """generates simulated data using the input params in config

        Args: 
            config {dict} -- keys required:
                model {str} -- Name of model to use to generate the data (in title case)
                starting_date {datetime} -- Starting date of the simulated data (in YYYY-MM-DD format)
                total_days {int} -- Number of days for which simulated data has to be generated (default: 50)
                initial_values {dict} -- Initial values for 'Active', 'Recovered' and 'Deceased' bucket
                params {dict} -- Parameters to generate the simulated data
        
        Returns:
            dict:
                data_frame {pd.DataFrame} -- dataframe of cases for a particular state, district with 5 columns : 
                    ['date', 'total', 'active', 'deceased', 'recovered']
                actual_params {dict} -- parameter values used to create the simulated data
        """
        if not config['fix_params']:
            for param in config['params']:
                if param == 'N':
                    continue
                config['params'][param] = getattr(np.random, config['params'][param][1])(config['params'][param][0][0], config['params'][param][0][1])
        actual_params = copy.deepcopy(config['params'])
        del actual_params['N']
        print ("parameters used to generate data:", actual_params)

        model_params = config['params']
        model_params['starting_date'] = config['starting_date']

        initial_colums = ['active', 'recovered', 'deceased']
        observed_values = {col : config['initial_values'][col] for col in initial_colums}
        observed_values['total'] = sum(observed_values.values())
        observed_values['date'] =  config['starting_date']
        model_params['observed_values'] = pd.DataFrame.from_dict([observed_values]).iloc[0,:]

        solver = eval(config['model'])(**model_params)
        if config['total_days']:
            df_result = solver.predict(total_days=config['total_days'])
        else:
            df_result = solver.predict()

        save_dir = '../../data/data/simulated_data/'    
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df_result.to_csv(os.path.join(save_dir, config['output_file_name']))
        pd.DataFrame([actual_params]).to_csv(os.path.join(save_dir, 'params_'+config['output_file_name']), index=False)
        return {"data_frame": df_result, "actual_params": actual_params}

    def pull_dataframes_cached(self, reload_data=False, label=None, **kwargs):
        return super().pull_dataframes_cached(reload_data=reload_data, label=label, **kwargs)

    def add_noise_to_df(self, df, noise_params):
        if 'active' in noise_params['columns_to_change']:
            columns_to_change = list(set(noise_params['columns_to_change'] + ['total', 'recovered', 'deceased']))
            columns_to_change.remove('active')
        else:
            columns_to_change = noise_params['columns_to_change']
        daily_cases = df[columns_to_change] - df[columns_to_change].shift(1)
        for col in columns_to_change:
            daily_cases[col] = pd.Series([0] + list(np.random.poisson(daily_cases[col].to_list()[1:])))
            daily_cases[col] = daily_cases[col].cumsum().add(df.loc[0, col])
        df[columns_to_change] = daily_cases[columns_to_change]
        if 'active' in noise_params['columns_to_change']:
            df['active'] = df['total'] - df['recovered'] - df['deceased']
        return df
    
    def get_data(self, **dataloading_params):
        """Main function serving as handshake between data and fitting modules

        Returns:
            dict{str : pd.DataFrame}: The processed dataframe
        """
        if dataloading_params['generate']:
            data_dict =  self.generate_data(**dataloading_params)
        else:
            data_dict = self.get_data_from_file(**dataloading_params)

        if dataloading_params['add_noise'] : 
            data_dict['data_frame'] = self.add_noise_to_df(data_dict['data_frame'],dataloading_params['noise'])
        return data_dict

    def generate_data(self, config_file, sim_data_configs_dir="../../configs/simulated_data/",
                      columns=['total', 'active', 'deceased', 'recovered'], **kwargs):
        """Generates simulated data using the input params in config file

        Args:
            configfile {str} -- Name of config file (located at '../../configs/simulated_data/') 
            required to generate the simulated data
        
        Returns:
            pd.DataFrame -- dataframe of cases for a particular state, district with 5 columns : 
                ['date', 'total', 'active', 'deceased', 'recovered'] and params used to generate data
        """

        with open(os.path.join(sim_data_configs_dir, config_file)) as configfile:
            config = yaml.load(configfile, Loader=yaml.SafeLoader)

        data_dict = self.pull_dataframes_cached(**config)
        df_result, params = data_dict['data_frame'], data_dict['actual_params']

        for col in df_result.columns:
            if col in columns:
                df_result[col] = df_result[col].astype('int64')
        return {"data_frame": df_result[['date'] + columns],
                "actual_params": params}

    def get_data_from_file(self, filename, params_filename=None,
                           columns=['total', 'active', 'deceased', 'recovered'], **kwargs):
        """Gets simulated data already generated in a file

        Args:
            filename (str): The filename containing the generated data
            params_filename (str, optional): Filename of params used to generate data. Defaults to None.
            columns (list, optional): List of columns to return. 
            Defaults to ['total', 'active', 'deceased', 'recovered'].

        Returns:
            dict{str : pd.DataFrame, str : params}: Processed dataframes, ideal params
        """
        params = {}
        if params_filename:
            params = pd.read_csv(params_filename).iloc[0, :].to_dict()
        df_result = pd.read_csv(filename)
        df_result['date'] = pd.to_datetime(df_result['date'])
        df_result.loc[:, columns] = df_result[columns].apply(pd.to_numeric)
        for col in df_result.columns:
            if col in columns:
                df_result[col] = df_result[col].astype('int64')
        return {"data_frame": df_result[['date' + columns]],
                "actual_params": params}
