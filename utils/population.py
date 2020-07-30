import numpy as np
import pandas as pd
import os


def create_age_bands(notation='-'):
    age_bands = {}
    for a in range(9):
        lower = a * 10
        upper = lower + 9
        if lower == 80:
            age_bands[f'{lower}+'] = list(range(lower, upper + 11))
            age_bands[f'{lower}+'] += ['100+']
        else:
            age_bands[f'{lower}-{upper}'] = list(range(lower, upper + 1))
    return age_bands


def get_india_district_age_bands(age_data):
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


def india_census_age():
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


def clean_indian_census(state):
    raw_all_district_age_data = india_census_age()[state]
    transposed = raw_all_district_age_data.head(3).T
    transposed.fillna('', inplace=True)
    new = transposed[transposed.columns[0]]
    for x in transposed.columns[1:]:
        new += transposed[x]
    all_district_age_data = raw_all_district_age_data.copy()
    all_district_age_data.columns = new.T
    return all_district_age_data[4:]


def get_district_population(state, area_names):
    all_district_age_data = clean_indian_census(state)
    district_age_data = all_district_age_data[all_district_age_data['Area Name'].isin(area_names)]
    total_pop = district_age_data[district_age_data['Age'] == 'All ages']['TotalPersons'].sum()
    return total_pop


def get_population(region, sub_region=None):
    population = pd.read_csv('../../data/data/population.csv')
    population = population[population['region'] == region.lower()]
    if sub_region is not None:
        population = population[population['sub_region'] == sub_region]
    return population.iloc[0]['population']


def get_country_age_bands(country):
    filename = '../../data/data/population_estimates_2017/IHME_GBD_2017_POP_2015_2017_Y2018M11D08 2.CSV'
    df = pd.read_csv(filename)
    countrydf = df[df['location_name'] == country]
    countrydf = df[df['year_id'] == 2017]

    age_bands = create_age_bands()
    age_band_pops = {}
    for key, band in age_bands.items():
        pop = countrydf[countrydf['age_group_name'].isin([str(b) for b in band])]['val'].sum()
        age_band_pops[key] = pop

    return age_band_pops, sum(age_band_pops.values())


def standardise_age(timeseries, country, state, area_names):
    data = timeseries.set_index('date')

    # country age bands/population
    ref_age_band_pops, country_total_pop = get_country_age_bands(country)
    assert (country_total_pop == sum(ref_age_band_pops.values()))
    ref_age_band_ratios = {k: v / country_total_pop for k, v in ref_age_band_pops.items()}

    # indian district age bands/population
    all_district_age_data = clean_indian_census(state)
    district_age_data = all_district_age_data[all_district_age_data['Area Name'].isin(area_names)]
    district_age_band_pops, district_total_pop = get_india_district_age_bands(district_age_data)
    assert (district_total_pop == sum(district_age_band_pops.values()))
    district_age_band_ratios = {k: v / district_total_pop for k, v in district_age_band_pops.items()}

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

    # mortality rate from country
    daily_observed_mortality = data['deceased'] / country_total_pop

    # how much different is it observed than 'implied'
    daily_mortality_ratio = daily_observed_mortality / implied_mortality

    # mortality rate accounting for observed/implied discrepancy
    age_stratified_daily_mortality = {k: daily_mortality_ratio * ref_mortality_rate[k] for k in
                                      ref_mortality_rate.keys()}
    # mortality rate accounting for observed/implied discrepancy and weighted by model location age
    age_std_mortality = sum(
        [age_stratified_daily_mortality[k] * district_age_band_ratios[k] for k in district_age_band_ratios.keys()])
    # print(age_std_mortality)

    checker = pd.DataFrame()
    checker['age_std_mortality'] = age_std_mortality
    checker['mortality'] = daily_observed_mortality
    checker.index.name = 'date'

    checker[f'log_age_std_mortality'] = checker['age_std_mortality'].apply(np.log)
    checker[f'log_mortality'] = checker['mortality'].apply(np.log)
    checker['day'] = [x for x in range(len(checker))]
    checker.loc[:, 'group'] = len(checker) * [1.0]
    checker.loc[:, 'covs'] = len(checker) * [1.0]

    return checker.reset_index(), country_total_pop
