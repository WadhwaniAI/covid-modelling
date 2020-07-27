from abc import abstractmethod

import numpy as np


class Uncertainty():
    def __init__(self, region_dict):
        self.region_dict = region_dict
        self.distribution = None
    
    @abstractmethod
    def get_distribution(self):
        pass

    @abstractmethod
    def get_forecasts(self, percentiles=None):
        pass

    def get_ptiles_idx(self, percentiles=None):
        """
        Get the predictions at certain percentiles from a distribution of trials

        Args:
            percentiles (list, optional): percentiles at which predictions from the distribution
                will be returned. Defaults to all deciles 10-90, as well as 2.5/97.5 and 5/95.

        Returns:
            dict: {percentile: index} where index is the trial index (to arrays in predictions_dict)
        """
        if self.distribution is None:
            raise Exception("No distribution found. Must call get_distribution first.")

        if percentiles is None:
            percentiles = range(10, 100, 10), np.array([2.5, 5, 95, 97.5])
            percentiles = np.sort(np.concatenate(percentiles))
        else:
            np.sort(percentiles)

        ptile_dict = {}
        for ptile in percentiles:
            index_value = (self.distribution['cdf'] - ptile / 100).apply(abs).idxmin()
            best_idx = self.distribution.loc[index_value - 2:index_value + 2, :]['idx'].min()
            ptile_dict[ptile] = int(best_idx)

        return ptile_dict
