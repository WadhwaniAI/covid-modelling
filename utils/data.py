import pandas as pd
import numpy as np
import requests
import datetime

# ---------FUNCTIONS FOR GETTING GLOBAL DATA----------

# helper function for modifying the dataframes such that each row is a snapshot of a country on a particular day
def _modify_dataframe(df, column_name='RecoveredCases'):
    cases_matrix = df.to_numpy()[:, 4:]
    cases_array = cases_matrix.reshape(-1, 1)

    province_info_matrix = df.to_numpy()[:, :4]
    province_info_array = np.repeat(province_info_matrix, cases_matrix.shape[1], axis=0)

    date = pd.to_datetime(df.columns[4:])
    date_array = np.tile(date, cases_matrix.shape[0]).reshape(-1, 1)

    data = np.concatenate((province_info_array, date_array, cases_array), axis=1)
    df = pd.DataFrame(data=data, columns=['Province/State', 'Country/Region', 'Lat', 'Long', 'Date', column_name])
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Loads time series case data for every country (and all provinces within certain countries) from JHU's github repo
def get_global_data():
    """
    This function parses the confirmed, death and recovered CSVs on JHU's github repo and converts them to pandas dataframes
    Columns of returned dataframe : 
    ['Province/State', 'Country/Region', 'Lat', 'Long', 'Date', 'ConfirmedCases', 'Deaths', 'RecoveredCases', 'ActiveCases']
    """
    df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    df_death = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    df_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

    df_confirmed = _modify_dataframe(df_confirmed, 'ConfirmedCases')
    df_death = _modify_dataframe(df_death, 'Deaths')
    df_recovered = _modify_dataframe(df_recovered, 'RecoveredCases')

    df_master = df_confirmed.merge(df_death, how='outer').merge(df_recovered, how='outer')
    df_master['ActiveCases'] = df_master['ConfirmedCases'] - df_master['Deaths'] - df_master['RecoveredCases']

    return df_master

# ---------FUNCTIONS FOR GETTING INDIAN DATA----------

def get_indian_data():
    """
    This function parses 5 JSONs : 4 JSONs from covid19india.org, and 1 JSON from rootnet.in
    It then converts the data into pandas dataframes
    It returns 7 dataframes : 
     - df_tested : Time series of people tested in India
     - df_statewise : Today's snapshot of cases in India, statewise
     - df_state_time_series : Time series of statewise cases in India
     - df_india_time_series : Time series of cases in India (nationwide)
     - df_districtwise : Today's snapshot of cases in India, districtwise (Unofficial)
     - df_raw_data : (Lots of) information about every patient (where they were detected, current status, 
       who all did they infect, etc). (Unofficial : from newsarticles, twitter, etc)
     - df_travel_history : Travel history of some patients (Unofficial : from newsarticles, twitter, etc)
    """
    # Parse data.json file
    data = requests.get('https://api.covid19india.org/data.json').json()

    # Create dataframe for testing data
    df_tested = pd.DataFrame.from_dict(data['tested'])
    df_tested = df_tested[np.logical_not(df_tested['source'] == '')]
    df_tested.reset_index(inplace=True, drop=True)

    # Create dataframe for statewise data
    df_statewise = pd.DataFrame.from_dict(data['statewise'])

    # Create dataframe for time series data
    df_india_time_series = pd.DataFrame.from_dict(data['cases_time_series'])
    df_india_time_series['date'] = pd.to_datetime([datetime.datetime.strptime(x[:6]+ ' 2020', '%d %b %Y') for x in df_india_time_series['date']])
    
    # Parse state_district_wise.json file
    data = requests.get('https://api.covid19india.org/state_district_wise.json').json()
    states = data.keys()
    for state in states:
        for district, district_dict in data[state]['districtData'].items():
            delta_dict = dict([('delta_'+k, v) for k, v in district_dict['delta'].items()])
            data[state]['districtData'][district].update(delta_dict)
            del data[state]['districtData'][district]['delta']

    columns = ['state', 'district', 'confirmed', 'delta_confirmed', 'lastupdatedtime']
    df_districtwise = pd.DataFrame(columns=columns)
    for state in states:
        df = pd.DataFrame.from_dict(data[state]['districtData']).T.reset_index()
        df.columns = ['district', 'confirmed', 'delta_confirmed', 'lastupdatedtime']
        df['state'] = state
        df = df[columns]
        df_districtwise = pd.concat([df_districtwise, df], ignore_index=True)
        
    # Parse raw_data.json file
    # Create dataframe for raw history
    data = requests.get('https://api.covid19india.org/raw_data.json').json()
    df_raw_data = pd.DataFrame.from_dict(data['raw_data'])
    df_raw_data = df_raw_data[np.logical_not(df_raw_data['source1'] == '')]

    # Parse travel_history.json file
    # Create dataframe for travel history
    data = requests.get('https://api.covid19india.org/travel_history.json').json()
    df_travel_history = pd.DataFrame.from_dict(data['travel_history'])

    # Read states time series data from rootnet.in
    data = requests.get('https://api.rootnet.in/covid19-in/stats/history').json()
    columns = ['confirmedCasesForeign', 'confirmedCasesIndian', 'deaths', 'discharged', 'loc', 'date']
    df_state_time_series = pd.DataFrame(columns=columns)
    for i, data_json in enumerate(data['data']):
        df_temp = pd.DataFrame.from_dict(data['data'][i]['regional'])
        df_temp['date'] = data['data'][i]['day']
        df_state_time_series = pd.concat([df_state_time_series, df_temp])
        
    df_state_time_series['confirmedCases'] = df_state_time_series['confirmedCasesForeign'] + df_state_time_series['confirmedCasesIndian']
    df_state_time_series['date'] = pd.to_datetime(df_state_time_series['date'])
    df_state_time_series.columns = [x if x != 'loc' else 'state' for x in df_state_time_series.columns]
    
    return df_tested, df_statewise, df_state_time_series, df_india_time_series, df_districtwise, df_raw_data, df_travel_history