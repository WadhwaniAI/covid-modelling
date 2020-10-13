import pandas as pd
import numpy as np
import datetime
import requests

from data.dataloader.base import BaseLoader


class RootnetLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def load_data(self):
        """
        This function parses multiple JSONs from api.rootnet.in
        It then converts the data into pandas dataframes
        It returns the following dataframes as a dict : 
        - df_state_time_series : Time series of cases in India (statewise)
        - df_statewise_beds : Statewise bed capacity in India
        - df_medical_colleges : List of medical collegs in India
        """
        dataframes = {}
        
        # Read states time series data from rootnet.in
        data = requests.get('https://api.rootnet.in/covid19-in/stats/history').json()
        columns = ['confirmedCasesForeign', 'confirmedCasesIndian', 'deaths', 'discharged', 'loc', 'date']
        df_state_time_series = pd.DataFrame(columns=columns)
        for i, _ in enumerate(data['data']):
            df_temp = pd.DataFrame.from_dict(data['data'][i]['regional'])
            df_temp['date'] = data['data'][i]['day']
            df_state_time_series = pd.concat([df_state_time_series, df_temp])
            
        df_state_time_series['confirmedCases'] = df_state_time_series['confirmedCasesForeign'] + df_state_time_series['confirmedCasesIndian']
        df_state_time_series['date'] = pd.to_datetime(df_state_time_series['date'])
        df_state_time_series.columns = [x if x != 'loc' else 'state' for x in df_state_time_series.columns]
        del df_state_time_series['confirmedCasesIndian']
        del df_state_time_series['confirmedCasesForeign']
        del df_state_time_series['totalConfirmed']
        df_state_time_series['active'] = df_state_time_series['confirmedCases'] - \
            df_state_time_series['deaths'] - df_state_time_series['discharged']
        df_state_time_series.columns = ['deceased', 'recovered', 'state', 'date', 'total', 'active']
        df_state_time_series = df_state_time_series[['state', 'date', 'active', 'total', 'deceased', 'recovered']]
        dataframes['df_state_time_series'] = df_state_time_series
        
        data = requests.get('https://api.rootnet.in/covid19-in/hospitals/beds').json()
        df_statewise_beds = pd.DataFrame.from_dict(data['data']['regional'])
        dataframes['df_statewise_beds'] = df_statewise_beds

        data = requests.get('https://api.rootnet.in/covid19-in/hospitals/medical-colleges').json()
        df_medical_colleges = pd.DataFrame.from_dict(data['data']['medicalColleges'])
        dataframes['df_medical_colleges'] = df_medical_colleges

        return dataframes

    def get_rootnet_api_data(self):
        return self.load_data()

