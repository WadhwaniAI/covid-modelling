import sys
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pickle
import os

sys.path.append('../..')
from data.dataloader import get_covid19india_api_data, get_jhu_data
from data.processing import get_concat_data, get_data

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

def india_all():
	df = get_dataframes_cached()['df_india_time_series']
	df = dataframes['df_india_time_series']
	# df.dtypes
	df = df[['date', 'totalconfirmed', 'totaldeceased','totalrecovered']]
	df.columns = [df.columns[0]] + [col[5:] for col in df.columns[1:]]
	return df

def india_all_state():
	df = get_dataframes_cached()['df_india_time_series']
	df = dataframes['df_districts']
	df = df.groupby(['state', 'date']).sum()
	df.reset_index()
	return df

def india_state(state):
	return get_data(get_dataframes_cached(), state=state)

def india_states(states):
	dataframes = get_dataframes_cached()
	all_states = get_data(dataframes, state=states[0])
	for state in states[1:]:
		pd.concat(all_states, get_data(dataframes, state=state))
	return all_states

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
	
def get_dataframes_cached():
	picklefn = "../../cache/dataframes_ts_{today}.pkl".format(today=datetime.today().strftime("%d%m%Y"))
	try:
		print(picklefn)
		with open(picklefn, 'rb') as pickle_file:
			dataframes = pickle.load(pickle_file)
	except:
		print("pulling from source")
		dataframes = get_covid19india_api_data()
		if not os.path.exists('../../cache/'):
			os.mkdir('../../cache/')
		with open(picklefn, 'wb+') as pickle_file:
			pickle.dump(dataframes, pickle_file)
	return dataframes

def get_district_timeseries_cached(district, state, disable_tracker=False, filename=None, data_format='new'):
	picklefn = "../../cache/{district}_ts_{today}.pkl".format(
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
		# district_timeseries = get_concat_data(dataframes, state=state, district=district, new_district_name=new_district, concat=True)
		district_timeseries = get_data(dataframes, state=state, 
			district=district, disable_tracker=disable_tracker, 
			filename=filename, data_format=data_format)
		with open(picklefn, 'wb+') as pickle_file:
			pickle.dump(district_timeseries, pickle_file)
	return district_timeseries