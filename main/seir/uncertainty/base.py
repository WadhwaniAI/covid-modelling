from abc import abstractmethod

class Uncertainty():
    def __init__(self, predictions_dict):
        self.predictions_dict = predictions_dict
    
    @abstractmethod
    def get_distribution(self):
        pass

    @abstractmethod
    def get_forecasts(self, percentiles=None):
        pass
