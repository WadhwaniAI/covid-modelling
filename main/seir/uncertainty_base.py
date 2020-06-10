
import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
from functools import partial
from hyperopt import fmin, tpe, hp, Trials

sys.path.append('../../')
from main.seir.forecast import get_forecast
from main.seir.fitting import calculate_loss, train_val_split

from abc import abstractmethod

class Uncertainty():
    def __init__(self, region_dict):
        self.region_dict = region_dict
    
    @abstractmethod
    def get_distribution(self):
        pass

    @abstractmethod
    def get_forecasts(self, percentiles=None):
        pass
