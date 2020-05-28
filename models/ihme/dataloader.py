import sys
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pickle

sys.path.append('../..')
from data.dataloader import get_covid19india_api_data, get_rootnet_api_data, get_jhu_data
from data.processing import get_concat_data, get_district_time_series

def bbmp():
	df = pd.read_csv('../../data/data/bbmp.csv')
	df.loc[:, 'date'] = df['Date'].apply(lambda x: datetime.strptime(x, "%d.%m.%Y"))
	df.drop("Date", axis=1, inplace=True)
	df.loc[:, 'day'] = (df['date'] - df['date'].min()).dt.days
	# df.loc[:, 'day'] = pd.Series([i+1 for i in range(len(df))])
	df.rename(columns = {'Cumulative Deaths til Date':'deaths'}, inplace = True) 
	df.rename(columns = {'Cumulative Cases Til Date':'cases'}, inplace = True) 
	df.loc[:, 'group'] = pd.Series([1 for i in range(len(df))])
	return df

strs = [('deceased', 'deaths'), ('confirmed', 'cases'), ('recovered', 'recovered')]

def _covid19india():
	dfs = []
	for (name,_) in strs:
		df = pd.read_csv('http://api.covid19india.org/states_daily_csv/{}.csv'.format(name))
		df.dropna(axis=1, how='all', inplace=True)
		df.loc[:, 'date'] = df['date'].apply(lambda x: datetime.strptime(x, "%d-%b-%y"))
		df.columns = [df.columns[0]] + [colname + '_{}'.format(name) for colname in df.columns[1:]]
		dfs.append(df)

	result = dfs[0]
	for df in dfs[1:]:
		result = result.merge(df, on='date', how='outer')
	result.loc[:, 'day'] = (result['date'] - result['date'].min()).dt.days
	# result.loc[:, 'day'] = pd.Series([i+1 for i in range(len(df))])
	result.set_index('date')
	return result

def india_all():
	df = _covid19india() 
	for (name, newname) in strs:
		df['TT_{}'.format(name)] = df['TT_{}'.format(name)].cumsum()
		df = df.rename(columns = {'TT_{}'.format(name):newname}, inplace = False) 
	df['state'] = 'TT'
	df = df[['date', 'day', 'state', 'deaths', 'cases', 'recovered']]
	return df

def india_all_state():
	dfs = []
	for (name, newname) in strs:
		df = pd.read_csv('http://api.covid19india.org/states_daily_csv/{}.csv'.format(name))
		df.dropna(axis=1, how='all', inplace=True)
		df.loc[:, 'date'] = df['date'].apply(lambda x: datetime.strptime(x, "%d-%b-%y"))
		df.loc[:, 'day'] = (df['date'] - df['date'].min()).dt.days
		# df.loc[:, 'day'] = pd.Series([i+1 for i in range(len(df))])
		df = df.set_index(['date', 'day'])
		df = df.stack().reset_index()
		df.columns = ['date', 'day', 'state', newname]
		dfs.append(df)

	result = dfs[0]
	for df in dfs[1:]:
		result = result.merge(df, on=['date', 'state', 'day'], how='outer')

	return result

def india_state(state):
	df = india_all_state()
	state = df.loc[df['state'] == state].reset_index()
	for (_, newname) in strs:
		state[newname] = state[newname].cumsum()
	return state

def india_states(states):
	df = india_all_state()
	state = df[df['state'].isin(states)].reset_index()
	for (_, newname) in strs:
		state[newname] = state[newname].cumsum()
	return state

def jhu(country):
	df_master = get_jhu_data()
	# print(df_master['Country/Region'].unique())
	# Province/State            object
	# Country/Region            object
	# Lat                      float64
	# Long                     float64
	# Date              datetime64[ns]
	# ConfirmedCases            object
	# Deaths                    object
	# RecoveredCases            object
	# ActiveCases               object
	df = df_master[df_master['Country/Region'] == country]
	df.loc[:, 'day'] = (df['Date'] - df['Date'].min()).dt.days
	df.loc[:, 'confirmed'] = pd.to_numeric(df['ConfirmedCases'])
	df.loc[:, 'deceased'] = pd.to_numeric(df['Deaths'])
	df.loc[:, 'recovered'] = pd.to_numeric(df['RecoveredCases'])
	df.loc[:, 'active'] = pd.to_numeric(df['ActiveCases'])
	df['Province/State'].fillna(country, inplace=True)
	df.columns = [c.lower() for c in df.columns]
	return df

def districtwise(district_state_tuple):
	district, state = district_state_tuple[0], district_state_tuple[1]
	today = datetime.today().strftime('%Y%m%d')
	filename = f'data/{district}_{today}.csv'
	try:
		districtdf = pd.read_csv(filename)
	except:
		print("Didnt find CSV, pulling from source")
		dataframes = get_covid19india_api_data()
		districtdf = get_district_time_series(dataframes, state=state, district=district)
		districtdf.to_csv(filename, index=False)

	districtdf.loc[:, 'date'] = pd.to_datetime(districtdf['date'])
	districtdf.columns = ['date', 'cases', 'deaths']
	districtdf.loc[:, 'group'] = district
	districtdf.loc[:, 'day'] = (districtdf['date'] - districtdf['date'].min()).dt.days
	
	return districtdf
	
def get_dataframes_cached():
	picklefn = "data/dataframes_ts_{today}.pkl".format(today=datetime.today().strftime("%d%m%Y"))
	try:
		print(picklefn)
		with open(picklefn, 'rb') as pickle_file:
			dataframes = pickle.load(pickle_file)
	except:
		print("pulling from source")
		dataframes = get_covid19india_api_data()
		with open(picklefn, 'wb+') as pickle_file:
			pickle.dump(dataframes, pickle_file)
	return dataframes

def get_district_timeseries_cached(district, state):
	picklefn = "data/{district}_ts_{today}.pkl".format(
		district=district, today=datetime.today().strftime("%d%m%Y")
	)	
	try:
		print(picklefn)
		with open(picklefn, 'rb') as pickle_file:
			district_timeseries = pickle.load(pickle_file)
	except:
		new_district = district
		if district == 'Bengaluru':
			district = ['Bengaluru Urban', 'Bengaluru Rural']
		elif district == 'Delhi':
			district = ['East Delhi', 'New Delhi', 'North Delhi', 
			'North East Delhi','North West Delhi', 'South Delhi', 
			'South West Delhi','West Delhi', 'Unknown', 'Central Delhi', 
			'Shahdara','South East Delhi','']
		dataframes = get_dataframes_cached()
		district_timeseries = get_concat_data(dataframes, state=state, district=district, new_district_name=new_district, concat=True)
		with open(picklefn, 'wb+') as pickle_file:
			pickle.dump(district_timeseries, pickle_file)
	return district_timeseries