import pandas as pd
import numpy as np
import os
import yaml
import copy
from models.seir import *

from data.dataloader.base import BaseLoader

class SimulatedDataLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def pull_dataframes(self, **config):
        """generates simulated data using the input params in config
        Keyword Arguments
        -----------------
            config {dict} -- keys required:
                model {str} -- Name of model to use to generate the data (in title case)
                starting_date {datetime} -- Starting date of the simulated data (in YYYY-MM-DD format)
                total_days {int} -- Number of days for which simulated data has to be generated (default: 50)
                initial_values {dict} -- Initial values for 'Active', 'Recovered' and 'Deceased' bucket
                params {dict} -- Parameters to generate the simulated data
        
        Returns
        -------
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

    def get_data(self, **dataloading_params):
        if dataloading_params['generate']:
            return self.generate_data(**dataloading_params)
        else:
            return self.get_data_from_file(**dataloading_params)

    def generate_data(self, config_file, generate=True, sim_data_configs_dir="../../configs/simulated_data/",
                      columns=['total', 'active', 'deceased', 'recovered'], **kwargs):
        """generates simulated data using the input params in config file
        Keyword Arguments
        -----------------
            configfile {str} -- Name of config file (located at '../../configs/simulated_data/') 
            required to generate the simulated data
        
        Returns
        -------
            pd.DataFrame -- dataframe of cases for a particular state, district with 5 columns : 
                ['date', 'total', 'active', 'deceased', 'recovered']
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

    def get_data_from_file(self, filename, params_filename=None, generate=False, 
                           columns=['total', 'active', 'deceased', 'recovered'], **kwargs):
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
