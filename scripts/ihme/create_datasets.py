"""
Script to create IHME generated synthetic dataset
"""
import argparse
import datetime
import os
import sys

import pandas as pd
import yaml

sys.path.append('../../')

from data.processing import get_data


def create_datasets(input_path, output_path, synthetic_days):
    folders = [f.name for f in os.scandir(input_path) if f.is_dir()]
    for folder in folders:
        folder_path = os.path.join(input_path, folder)
        with open(os.path.join(folder_path, 'config.yaml')) as configfile:
            config = yaml.load(configfile, Loader=yaml.SafeLoader)

        # Get actual data
        original_data = get_data(config['fitting']['data']['data_source'],
                                 config['fitting'][
                                     'data']['dataloading_params'])

        # Determine when real data ends
        end_date = config['fitting']['split']['end_date']
        if end_date is None:
            start_date = config['fitting']['split']['start_date']
            train_period = config['fitting']['split']['train_period']
            val_period = config['fitting']['split']['val_period']
            test_period = config['fitting']['split']['test_period']
            end_date = start_date + datetime.timedelta(train_period + val_period +
                                                       test_period)
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

        # Get IHME predictions
        predictions = pd.read_csv(os.path.join(folder_path, 'df_prediction.csv'))
        predictions['date'] = pd.to_datetime(predictions['date'])

        # Combine datasets
        actual_data = original_data['data_frame']
        actual_data_before = actual_data[actual_data['date'] <= end_date]
        actual_data_after = actual_data[actual_data['date'] > end_date +
                                        datetime.timedelta(synthetic_days)]
        synthetic_data = predictions[predictions['date'] > end_date].reset_index().iloc[
                         :synthetic_days, :]
        synthetic_data = synthetic_data[
            ['date'] + config['fitting']['loss']['loss_compartments']]
        dataset = pd.concat([actual_data_before, synthetic_data, actual_data_after])
        dataset.reset_index(inplace=True)

        output_folder = os.path.join(output_path, (end_date + datetime.timedelta(synthetic_days)).strftime('%Y-%m-%d'))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        dataset.to_csv(os.path.join(output_folder, 'data.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help="input folder path", required=True)
    parser.add_argument("-o", "--output_path", help="output folder path", required=True)
    parser.add_argument("-n", "--synthetic_days", help="number of days of synthetic data used", type=int, required=True)
    args = parser.parse_args()

    create_datasets(args.input_path, args.output_path, args.synthetic_days)
