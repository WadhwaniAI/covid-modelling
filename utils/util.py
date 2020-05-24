import os
import sys
import numpy as np

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
    return series.rolling(window).mean()