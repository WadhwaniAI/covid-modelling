import pandas as pd
import numpy as np
import os
import copy
from datetime import timedelta
from models.seir import *

from data.dataloader.base import BaseLoader

class SimulatedDataLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def load_data(self, **config):
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
            pd.DataFrame -- dataframe of cases for a particular state, district with 5 columns : 
                ['date', 'total', 'active', 'deceased', 'recovered']
            dict -- parameter values used to create the simulated data
        """
        if config['set_seed']:
            np.random.seed(config['seed'])
        if (not config['fix_params']):
            for param in config['params']:
                if param == 'N':
                    continue
                config['params'][param] = getattr(np.random, config['params'][param][1])(
                    config['params'][param][0][0], config['params'][param][0][1])
        actual_params = copy.deepcopy(config['params'])
        del actual_params['N']
        print ("parameters used to generate data:", actual_params)

        model_params = config['params']
        model_params['starting_date'] = config['starting_date']

        initial_colums = ['active', 'recovered', 'deceased']
        for key in initial_colums:
            if isinstance(config['initial_values'][key], list):
                config['initial_values'][key] = \
                    np.random.randint(config['initial_values'][key][0][0],
                                      config['initial_values'][key][0][1])
        print("Initial values used to generate data:", config['initial_values'])

        observed_values = {col : config['initial_values'][col] for col in initial_colums}
        observed_values['total'] = sum(observed_values.values())
        observed_values['date'] =  config['starting_date']
        model_params['observed_values'] = pd.DataFrame.from_dict([observed_values]).iloc[0,:]

        solver = eval(config['model'])(**model_params)
        if (config['total_days']):
            df_result = solver.predict(total_days=config['total_days'])
        else:
            df_result = solver.predict()

        if config['save']:
            save_dir = '../../data/data/simulated_data/'
            os.makedirs(save_dir, exist_ok=True)
            df_result.to_csv(os.path.join(save_dir, config['output_file_name']))
            pd.DataFrame([actual_params]).to_csv(
                f'{save_dir}params_{config["output_file_name"]}', index=False)
        return {"data_frame": df_result, "actual_params": actual_params} 

    def simulate_spike(self, df, comp, start_date, end_date, frac_to_report):
        df_spiked = copy.deepcopy(df)
        df_diff = copy.deepcopy(df)
        # Create incident cases dataframe
        num_cols = df_diff.select_dtypes(include='number').columns
        df_diff.loc[:, num_cols] = df_diff.loc[:, num_cols].diff()
        # Filter the dataframe that has to be spiked
        df_to_spike = df_diff[(df_diff['date'].dt.date >= start_date)
                              & (df_diff['date'].dt.date <= end_date)]
        # Modifiying time frame dfs to report only fraction of cases
        daily_frac_to_report = np.random.beta(2, 6, size=len(df_to_spike))
        spike = np.dot(df_to_spike[comp], 1-daily_frac_to_report)
        df_to_spike.loc[:, 'active'] = df_to_spike.loc[:, 'active'] + \
            np.multiply(df_to_spike[comp], 1-daily_frac_to_report)
        df_to_spike.loc[:, comp] = df_to_spike.loc[:, comp] - \
            np.multiply(df_to_spike[comp], 1-daily_frac_to_report)

        # Converting df_to_spike from incident to cumsum array
        base_nums = df_spiked.loc[df_spiked['date'].dt.date == (start_date - timedelta(days=1)), num_cols] 
        df_to_spike.loc[:, num_cols] = df_to_spike.loc[:, num_cols].cumsum() 
        df_to_spike.loc[:, num_cols] += base_nums.to_numpy()

        df_spiked.loc[df_spiked['date'].isin(df_to_spike['date']), :] = df_to_spike

        df_spiked.loc[df_spiked['date'].dt.date == end_date, comp] += spike
        df_spiked.loc[df_spiked['date'].dt.date == end_date, 'active'] -= spike

        return df_spiked
