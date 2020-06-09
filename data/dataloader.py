import pandas as pd
import numpy as np
import requests
import datetime

from pyathena import connect
from pyathena.pandas_cursor import PandasCursor

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
def get_jhu_data():
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

# ---------FUNCTIONS FOR GETTING CUSTOM DATA----------

def create_connection(pyathena_rc_path=None):
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


def get_athena_dataframes(pyathena_rc_path=None):
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
    cursor = create_connection(pyathena_rc_path)

    # Run SQL SELECT queries to get all the tables in the database as pandas dataframes
    dataframes = {}
    tables_list = cursor.execute('Show tables').as_pandas().to_numpy().reshape(-1, )
    for table in tables_list:
        dataframes[table] = cursor.execute(
            'SELECT * FROM {}'.format(table)).as_pandas()
    
    return dataframes


# ---------FUNCTIONS FOR GETTING INDIAN DATA----------

def get_covid19india_api_data():
    """
    This function parses multiple JSONs from covid19india.org
    It then converts the data into pandas dataframes
    It returns the following dataframes as a dict : 
     - df_tested : Time series of people tested in India
     - df_statewise : Today's snapshot of cases in India, statewise
     - df_india_time_series : Time series of cases in India (nationwide)
     - df_districtwise : Today's snapshot of cases in India, districtwise
     - df_raw_data : Patient level information of cases
     - df_deaths_recoveries : Patient level information of deaths and recoveries
     - [NOT UPDATED ANYMORE] df_travel_history : Travel history of some patients (Unofficial : from newsarticles, twitter, etc)
     - df_resources : Repository of testing labs, fundraising orgs, government helplines, etc
    """

    # List of dataframes to return
    dataframes = {}

    # Parse data.json file
    data = requests.get('https://api.covid19india.org/data.json').json()

    # Create dataframe for testing data
    df_tested = pd.DataFrame.from_dict(data['tested'])
    dataframes['df_tested'] = df_tested

    # Create dataframe for statewise data
    df_statewise = pd.DataFrame.from_dict(data['statewise'])
    dataframes['df_statewise'] = df_statewise

    # Create dataframe for time series data
    df_india_time_series = pd.DataFrame.from_dict(data['cases_time_series'])
    df_india_time_series['date'] = pd.to_datetime([datetime.datetime.strptime(x[:6]+ ' 2020', '%d %b %Y') for x in df_india_time_series['date']])
    dataframes['df_india_time_series'] = df_india_time_series

    # Parse state_district_wise.json file
    data = requests.get('https://api.covid19india.org/state_district_wise.json').json()
    states = data.keys()
    for state in states:
        for district, district_dict in data[state]['districtData'].items():
            delta_dict = dict([('delta_'+k, v) for k, v in district_dict['delta'].items()])
            data[state]['districtData'][district].update(delta_dict)
            del data[state]['districtData'][district]['delta']

    columns = ['state', 'district', 'active', 'confirmed', 'deceased', 'recovered', 'delta_confirmed', 'delta_deceased', 'delta_recovered']
    df_districtwise = pd.DataFrame(columns=columns)
    for state in states:
        df = pd.DataFrame.from_dict(data[state]['districtData']).T.reset_index()
        del df['notes']
        df.columns = columns[1:]
        df['state'] = state
        df = df[columns]
        df_districtwise = pd.concat([df_districtwise, df], ignore_index=True)
    dataframes['df_districtwise'] = df_districtwise
        
    # Parse raw_data.json file
    # Create dataframe for raw history
    data = requests.get('https://api.covid19india.org/raw_data.json').json()
    df_raw_data_old = pd.DataFrame.from_dict(data['raw_data'])
    dataframes['df_raw_data_old'] = df_raw_data_old

    data = requests.get('https://api.covid19india.org/raw_data1.json').json()
    df_raw_data_1 = pd.DataFrame.from_dict(data['raw_data'])

    data = requests.get('https://api.covid19india.org/raw_data2.json').json()
    df_raw_data_2 = pd.DataFrame.from_dict(data['raw_data'])

    data = requests.get('https://api.covid19india.org/raw_data3.json').json()
    df_raw_data_3 = pd.DataFrame.from_dict(data['raw_data'])

    data = requests.get('https://api.covid19india.org/raw_data4.json').json()
    df_raw_data_4 = pd.DataFrame.from_dict(data['raw_data'])

    data = requests.get('https://api.covid19india.org/raw_data5.json').json()
    df_raw_data_5 = pd.DataFrame.from_dict(data['raw_data'])

    dataframes['df_raw_data'] = pd.concat(
        [df_raw_data_1, df_raw_data_2, df_raw_data_3, df_raw_data_4, df_raw_data_5], ignore_index=True)

    # Parse deaths_recoveries.json file
    data = requests.get('https://api.covid19india.org/deaths_recoveries.json').json()
    df_raw_data_2 = pd.DataFrame.from_dict(data['deaths_recoveries'])
    dataframes['df_deaths_recoveries'] = df_raw_data_2

    data = requests.get('https://api.covid19india.org/districts_daily.json').json()

    df_districts = pd.DataFrame(columns=['notes', 'active', 'confirmed', 'deceased', 'recovered', 'date', 'state', 'district'])
    for state in data['districtsDaily'].keys():
        for dist in data['districtsDaily'][state].keys():
            df = pd.DataFrame.from_dict(data['districtsDaily'][state][dist])
            df['state'] = state
            df['district'] = dist
            df_districts = pd.concat([df_districts, df], ignore_index=True)
            
    df_districts = df_districts[['state', 'district', 'date', 'active', 'confirmed', 'deceased', 'recovered', 'notes']]
    df_districts.loc[df_districts['district'] == 'Bengaluru', 'district'] = 'Bengaluru Urban'
    df_districts.loc[df_districts['district'] == 'Ahmadabad', 'district'] = 'Ahmedabad'
    
    numeric_cols = ['active', 'confirmed', 'deceased', 'recovered']
    df_districts[numeric_cols] = df_districts[numeric_cols].apply(pd.to_numeric)
    dataframes['df_districts'] = df_districts

    # Parse travel_history.json file
    # Create dataframe for travel history
    """
    !!!! TRAVEL HISTORY HAS BEEN DEPRECATED !!!!
    """
    data = requests.get('https://api.covid19india.org/travel_history.json').json()
    df_travel_history = pd.DataFrame.from_dict(data['travel_history'])
    dataframes['df_travel_history'] = df_travel_history

    data = requests.get('https://api.covid19india.org/resources/resources.json').json()
    df_resources = pd.DataFrame.from_dict(data['resources'])
    dataframes['df_resources'] = df_resources

    return dataframes

def get_rootnet_api_data():
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
    df_state_time_series['hospitalised'] = df_state_time_series['confirmedCases'] - \
        df_state_time_series['deaths'] - df_state_time_series['discharged']
    df_state_time_series.columns = ['deceased', 'recovered', 'state', 'date', 'total_infected', 'hospitalised']
    df_state_time_series = df_state_time_series[['state', 'date', 'hospitalised', 'total_infected', 'deceased', 'recovered']]
    dataframes['df_state_time_series'] = df_state_time_series
    
    data = requests.get('https://api.rootnet.in/covid19-in/hospitals/beds').json()
    df_statewise_beds = pd.DataFrame.from_dict(data['data']['regional'])
    dataframes['df_statewise_beds'] = df_statewise_beds

    data = requests.get('https://api.rootnet.in/covid19-in/hospitals/medical-colleges').json()
    df_medical_colleges = pd.DataFrame.from_dict(data['data']['medicalColleges'])
    dataframes['df_medical_colleges'] = df_medical_colleges

    return dataframes
