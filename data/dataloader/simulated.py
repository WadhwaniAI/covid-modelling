import pandas as pd
import numpy as np
import os
import copy
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
        ideal_params = {}
        if (not config['fix_params']):
            for param in config['params']:
                if param == 'N':
                    continue
                config['params'][param] = getattr(np.random, config['params'][param][1])(config['params'][param][0][0], config['params'][param][0][1])
                ideal_params[param] = config['params'][param]

        ideal_params = copy.deepcopy(config['params'])
        del ideal_params['N']
        print ("parameters used to generate data:", ideal_params)

        model_params = config['params']
        model_params['starting_date'] = config['starting_date']

        initial_colums = ['active', 'recovered', 'deceased']
        observed_values = {col : config['initial_values'][col] for col in initial_colums}
        observed_values['total'] = sum(observed_values.values())
        observed_values['date'] =  config['starting_date']
        model_params['observed_values'] = pd.DataFrame.from_dict([observed_values]).iloc[0,:]
        
        solver = eval(config['model'])(**model_params)
        if (config['total_days']):
            df_result = solver.predict(total_days=config['total_days'])
        else:
            df_result = solver.predict()

        save_dir = '../../data/data/simulated_data/'    
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df_result.to_csv(os.path.join(save_dir, config['output_file_name']))
        pd.Series((ideal_params)).to_csv(os.path.join(save_dir, 'params_'+config['output_file_name']))
        return df_result, ideal_params