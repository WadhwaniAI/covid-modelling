import os
import sys
import numpy as np
import yaml

class HidePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def train_test_split(df, threshold, threshold_col='date'):
            return df[df[threshold_col] <= threshold], df[df[threshold_col] > threshold]

def smooth(y, smoothing_window):
    box = np.ones(smoothing_window)/smoothing_window
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def rollingavg(series, window):
    return series.rolling(window, center=True).mean()

def read_config(path, backtesting=False):
    default_path = os.path.join(os.path.dirname(path), 'default.yaml')
    with open(default_path) as default:
        config = yaml.load(default, Loader=yaml.SafeLoader)
    with open(path) as configfile:
        new = yaml.load(configfile, Loader=yaml.SafeLoader)
    for k in config.keys():
        if type(config[k]) is dict and new.get(k) is not None:
            config[k].update(new[k])
    model_params = config['model_params']
    if backtesting:
        config['base'].update(config['backtesting'])
    else:
        config['base'].update(config['run'])
    config = config['base']
    return config, model_params