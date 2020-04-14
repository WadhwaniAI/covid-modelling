import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def bbmp():
	df = pd.read_csv('../../data/data/bbmp.csv')
	df.loc[:, 'date'] = df['Date'].apply(lambda x: datetime.strptime(x, "%d.%m.%Y"))
	df.drop("Date", axis=1, inplace=True)
	df.loc[:, 'day'] = pd.Series([i+1 for i in range(len(df))])
	df.rename(columns = {'Cumulative Deaths til Date':'deaths'}, inplace = True) 
	df.rename(columns = {'Cumulative Cases Til Date':'cases'}, inplace = True) 
	df.loc[:, 'group'] = pd.Series([1 for i in range(len(df))])
	return df

strs = [('deceased', 'deaths'), ('confirmed', 'cases'), ('recovered', 'recovered')]

def _covid19india():
	dfs = []
	for (name,_) in strs:
		df = pd.read_csv('https://raw.githubusercontent.com/covid19india/api/master/states_daily_csv/{}.csv'.format(name))
		df.dropna(axis=1, how='all', inplace=True)
		df.loc[:, 'date'] = df['date'].apply(lambda x: datetime.strptime(x, "%d-%b-%y"))
		df.columns = [df.columns[0]] + [colname + '_{}'.format(name) for colname in df.columns[1:]]
		dfs.append(df)

	# result = pd.concat(dfs, axis=1, join='outer')
	result = dfs[0]
	for df in dfs[1:]:
		result = result.merge(df, on='date', how='outer')
	result.loc[:, 'day'] = pd.Series([i+1 for i in range(len(df))])
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
		df = pd.read_csv('https://raw.githubusercontent.com/covid19india/api/master/states_daily_csv/{}.csv'.format(name))
		df.dropna(axis=1, how='all', inplace=True)
		df.loc[:, 'date'] = df['date'].apply(lambda x: datetime.strptime(x, "%d-%b-%y"))
		df.loc[:, 'day'] = pd.Series([i+1 for i in range(len(df))])
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

