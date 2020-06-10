
import os
import sys
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
