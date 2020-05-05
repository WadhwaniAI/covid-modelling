#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import requests
import datetime
import os
import sys

sys.path.append('../..')
from data.dataloader import get_covid19india_api_data, get_rootnet_api_data

def get_district_death_df(df_raw_data_2, state, district):
    deceased = df_raw_data_2[df_raw_data_2['patientstatus'] == 'Deceased']
    statedf = deceased[deceased['state'] == state]
    unknown = statedf[statedf['district'] == '']
    print(f'{len(unknown)} deaths in {state} with unknown district')
    if district == 'Ahmadabad':
        print(f'adding {len(unknown)} deaths to Ahmadabad count')
        statedf['district'][statedf['district'] == ''] = district
        districtdf = statedf[statedf['district'] == district]
    elif state == 'Maharashtra':
        num_unknown = len(unknown)
        thirds = num_unknown//3
        print(f'adding {thirds} deaths to Mumbai/Pune each')
        print(len(statedf[statedf['district'] == district]))
        unknown_index = statedf['district'][statedf['district'] == ''].index
        statedf.loc[statedf[statedf['district'] == ''].index[:thirds]] = 'Mumbai'
        statedf.loc[statedf[statedf['district'] == ''].index[thirds:]] = 'Pune'
        print(len(statedf[statedf['district'] == district]))
        districtdf = statedf[statedf['district'] == district]
    elif state == 'Rajasthan':
        districtdf = statedf[statedf['district'] == district]
    elif state == 'Karnataka':
        districtdf = statedf[statedf['district'] == district]
    elif state == 'Delhi':
        districtdf = statedf[statedf['district'] == district]
    return districtdf

def get_district_time_series(dataframes, state='Karnataka', district='Bengaluru'):
    if district == 'all' or type(district) == list:
        if district == 'all':
            districtwise = dataframes['df_districtwise']
            district = districtwise[districtwise['state'] == state]['district'].unique()
            state = len(district) * [state]
        district_timeseries = {}
        for (s, d) in list(zip(state, district)):
            district_timeseries[d] = get_district_time_series(dataframes, state=s, district=d)
        return district_timeseries
    else:
        df_raw_data_1 = dataframes['df_raw_data'][dataframes['df_raw_data']['detectedstate'] == state]
        df_raw_data_1 = df_raw_data_1[df_raw_data_1['detecteddistrict'] == district]
        df_raw_data_1['dateannounced'] = pd.to_datetime(df_raw_data_1['dateannounced'], format='%d/%m/%Y')
        index = pd.date_range(np.min(df_raw_data_1['dateannounced']), np.max(df_raw_data_1['dateannounced']))
        df_district = pd.DataFrame(columns=['total_confirmed'], index=index)
        df_district['total_confirmed'] = [0]*len(index)
        for _, row in df_raw_data_1.iterrows():
            df_district.loc[row['dateannounced']:, 'total_confirmed'] += 1

        # Deaths calculation
        deathsdf = get_district_death_df(dataframes['df_raw_data_2'], state, district)
        deathsdf = deathsdf[deathsdf['state'] == state]
        deathsdf = deathsdf[deathsdf['district'] == district]
        deathsdf['date'] = pd.to_datetime(deathsdf['date'], format='%d/%m/%Y')

        df_district['total_deaths'] = [0]*len(index)
        for _, row in deathsdf.iterrows():
            if row['patientstatus'] == 'Deceased':
                date = pd.to_datetime(row['date'], format='%d/%m/%Y')
                df_district.loc[date:, 'total_deaths'] += 1


        df_district.reset_index(inplace=True)
        df_district.columns = ['date', 'total_confirmed', 'total_deaths']
        return df_district

def clean(raw_all_district_age_data):
    transposed = raw_all_district_age_data.head(3).T
    transposed.fillna('', inplace=True)
    new = transposed[transposed.columns[0]]
    for x in transposed.columns[1:]:
        new += transposed[x] 
    all_district_age_data = raw_all_district_age_data.copy()
    all_district_age_data.columns = new.T
    return all_district_age_data[4:]

def create_age_bands():
    age_bands = {}
    for a in range(9):
        lower = a * 10
        upper = lower + 9
        if lower == 80:
            age_bands[f'{lower}+'] = list(range(lower, upper+11))
            age_bands[f'{lower}+'] += ['100+']
        else:
            age_bands[f'{lower}-{upper}'] = list(range(lower, upper+1))
    return age_bands

def get_age_band_population(age_data):
    # filter for age bands and sum
    total_pop = age_data[age_data['Age'] == 'All ages']['TotalPersons'].sum()
    unknown_pop = age_data[age_data['Age'] == 'Age not stated']['TotalPersons'].sum()
    total_pop -= unknown_pop
    age_bands = create_age_bands()

    age_band_pops = {}
    for key, band in age_bands.items():
        pop = age_data[age_data['Age'].isin(band)]['TotalPersons'].sum()
        age_band_pops[key] = pop
    return age_band_pops, total_pop

def census_age():
	# get age data in bands
	filenames = 'DDW-{}00C-13'
	state_file_mapping = {
		filenames.format('24'): 'Gujarat',
		filenames.format('29'): 'Karnataka',
		filenames.format('27'): 'Maharashtra',
		filenames.format('07'): 'Delhi',
		filenames.format('08'): 'Rajasthan',
	}

	age_data = {}
	directory = '../../data/data/census/'
	for filename in os.listdir(directory):
		df = pd.read_excel(os.path.join(directory, filename))
		age_data[state_file_mapping[filename.split('.')[0]]] = df.dropna(how='all')
	return age_data

def standardise_age(district_timeseries, age_data, district, state, area_names):
    data = district_timeseries.set_index('date')
    raw_all_district_age_data = age_data[state]

    all_district_age_data = clean(raw_all_district_age_data)

    # Get relevant district(s) data
    district_age_data = all_district_age_data[all_district_age_data['Area Name'].isin(area_names)]
    
    district_age_band_pops, district_total_pop = get_age_band_population(district_age_data)
    assert(district_total_pop == sum(district_age_band_pops.values()))

    district_age_band_ratios = {k: v / district_total_pop for k, v in district_age_band_pops.items()}
    print("dtp: {}".format(district_total_pop))
    ref_age_band_ratios = {
        '0-9': 0.1144545912,
        '10-19': 0.1096780507,
        '20-29': 0.1387701325,
        '30-39': 0.1481915984,
        '40-49': 0.1548679659,
        '50-59': 0.1428622446,
        '60-69': 0.1092853481,
        '70-79': 0.05542319854,
        '80+': 0.02646687006,
    }

    # calculated here: https://docs.google.com/spreadsheets/d/1APX7XwoJPIbUXOgXa2vZNreDn6UseBucSGq9fh-N5jE/edit#gid=1859974541

    # mortality, no sk
    ref_mortality_rate = {
        '0-9': 0.00000001922452747,
        '10-19': 0.00000001996437158,
        '20-29': 0.0000001540307269,
        '30-39': 0.0000004222770726,
        '40-49': 0.00000128438119,
        '50-59': 0.000005125162691,
        '60-69': 0.00001948507013,
        '70-79': 0.00009988464688,
        '80+': 0.0003878624777,
    }

    # mortality based on other countries
    implied_mortality = sum([ref_age_band_ratios[k] * ref_mortality_rate[k] for k in ref_mortality_rate.keys()])

    # need mortality rate from India -- this is going to be dependent on case data, so introducing that testing bias...
    # observed deaths/known cases
    # daily_observed_mortality = data['total_deaths']/data['total_confirmed']
    daily_observed_mortality = data['total_deaths']/district_total_pop

    # how much different is it observed than 'implied'
    daily_mortality_ratio = daily_observed_mortality/implied_mortality

    # mortality rate accounting for observed/implied discrepancy
    age_stratified_daily_mortality = {k: daily_mortality_ratio * ref_mortality_rate[k] for k in ref_mortality_rate.keys()}
    # mortality rate accounting for observed/implied discrepancy and weighted by model location age
    age_std_mortality = sum([age_stratified_daily_mortality[k] * district_age_band_ratios[k] for k in district_age_band_ratios.keys()])
    # print(age_std_mortality)

    checker = pd.DataFrame()
    # print(district_total_pop)
    checker['age_std'] = age_std_mortality
    checker['non_std'] = daily_observed_mortality
    # print(checker)
    return checker

if __name__ == "__main__":
    dataframes = get_covid19india_api_data()

    districts = ['Mumbai', 'Bengaluru', 'Ahmadabad', 'Jaipur', 'Pune', 'New Delhi']
    states = ['Maharashtra', 'Karnataka', 'Gujarat', 'Rajasthan', 'Maharashtra', 'Delhi']
    district_timeseries = get_district_time_series(dataframes, state=states, district=districts)

    # get age data in bands
    age_data = census_age()

    amd = 'District - Ahmadabad (07)'
    mumbai = 'District - Mumbai (23)'
    mumbai2 = 'District - Mumbai Suburban (22)'
    pune = 'District - Pune (25)'
    delhi ='District - New Delhi (05)'
    jaipur = 'District - Jaipur (12)'
    bengaluru = 'District - Bangalore (18)'

    df = standardise_age('Bengaluru', 'Karnataka', bengaluru)
    print(df)