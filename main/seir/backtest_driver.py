
import os
from datetime import datetime
import argparse
import pickle
import json
import matplotlib.pyplot as plt

import sys
sys.path.append('../..')
from data.dataloader import Covid19IndiaLoader()
from main.seir.fitting import get_regional_data
from main.seir.backtesting import SEIRBacktest

parser = argparse.ArgumentParser() 
parser.add_argument("-d", "--district", help="district name", required=True)
args = parser.parse_args()

print('\rgetting dataframes...')
loader = Covid19IndiaLoader()
dataframes = loader.get_covid19india_api_data()
state, district = 'Maharashtra', args.district.title()
fit = 'm1' # 'm1' or 'm2'
data_from_tracker=True

today = datetime.today()
output_folder = f'output/backtest/{district.lower()}/{today}'
if not os.path.exists(output_folder):
        os.makedirs(output_folder)

if district == 'Mumbai':
    data_format = 'old'
    filepath = '../../data/data/official-mumbai.csv'
elif district == 'Pune':
    data_format = 'new'
    filepath = '../../data/data/official-pune-21-05-20.csv'
else:
    data_format = None
    filepath = None

print('\rgetting district data...')
df_district, df_district_raw_data = get_regional_data(dataframes, state, district, 
                data_from_tracker, data_format, filepath)

print('starting backtesting...')
backtester = SEIRBacktest(state, district, df_district, df_district_raw_data, data_from_tracker)

results = backtester.test(fit, increment=3)
picklefn = f'{output_folder}/backtesting.pkl'

with open(picklefn, 'wb') as pickle_file:
        pickle.dump(results, pickle_file)

backtester.plot_results(which_compartments=['total_infected', 'deceased', 'hospitalised', 'recovered'])
plt.savefig('f{output_folder}/backtest.png')
plt.clf()

comp = 'deceased'
backtester.plot_errors(compartment=comp)
plt.savefig(f'{output_folder}/errors-{comp}.png')
plt.clf()

params = {
    'state': state,
    'district': district,
    'fit': fit,
    'data_from_tracker': data_from_tracker,
    'data_format': data_format,
    'filepath': filepath,
}
with open(f'{output_folder}/params.json', 'w') as pfile:
        json.dump(params, pfile)
print(f'see output at: {output_folder}')
