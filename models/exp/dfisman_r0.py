import pandas as pd
import numpy as np
from numpy import exp

def load_data(from_source=True):
	if not from_source:
		try:
			states = pd.read_csv('data/states.csv')
			counties = pd.read_csv('data/counties.csv')
			return states, counties
		except:
			print("Failed to load pickles, reading from source")
	
	# date,state,fips,cases,deaths
	states = pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv", header=0)

	# date,county,state,fips,cases,deaths
	counties = pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv", header=0)

	return states, counties

def clean(df):
	df["date"] = pd.to_datetime(df["date"])

def save_pickles(df_name_pairs):
	for df, name in df_name_pairs:
		filename = 'data/{}.pkl'.format(name)
		df.to_pickle(filename)
		print("Saved to {}".format(filename))

def save_csv(df_name_pairs):
	for df, name in df_name_pairs:
		filename = 'data/{}.csv'.format(name)
		df.to_csv(filename, index=False)
		print("Saved to {}".format(filename))

def calc_stats(df): # to use this date must be ordered by datetime, restricted to 1 area
	df["cfr"] = df["deaths"]/df["cases"]
	df["prev_day_cases"] = df["cases"].shift(1)
	df["r"] = df["cases"]/df["prev_day_cases"]
	df["delta_c"] = df["cases"] - df["prev_day_cases"]
	df["Rt"] = (7*(df["r"] - 1)).apply(exp)

def smooth_Rt(df):
	df["n_plus_1_cases"] = df["cases"].shift(-1)
	df["n_plus_2_cases"] = df["cases"].shift(-2)
	df["n_minus_2_cases"] = df["cases"].shift(2)
	df["n_minus_3_cases"] = df["cases"].shift(3)
	df["n_minus_4_cases"] = df["cases"].shift(4)
	df["n_minus_5_cases"] = df["cases"].shift(5)
	df["r_smooth"] = np.mean([df["n_plus_2_cases"], df["n_plus_1_cases"], df["cases"], df["prev_day_cases"], df["n_minus_2_cases"]])
	df["Rt_smooth"] = (7*(df["r_smooth"] - 1)).apply(exp)


def get_state(df, state):
	return df[df["state"] == state]


if __name__ == '__main__':
	load_from_source=False
	states, counties = load_data(from_source=load_from_source)
	
	if load_from_source:
		clean(states)
		clean(counties)
		save_csv([(states, "states"), (counties, "counties")])
	
	ny = get_state(states, "New York")
	ny = ny.sort_values(by='date',ascending=True)
	calc_stats(ny)
	smooth_Rt(ny)
	save_csv([(ny, "ny_smoothed")])
