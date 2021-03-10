
import pandas as pd

from data.dataloader.base import BaseLoader

class FileLoader(BaseLoader):
    """A very barebones dataloader for loading data from files.

    #TODO : Populate more + add checks

    Args:
        BaseLoader (abstract class): Abstract Data Loader Class
    """
    def __init__(self):
        pass

    def pull_dataframes(self, **kwargs):
        return {}

    def pull_dataframes_cached(self, reload_data=False, label=None, **kwargs):
        return super().pull_dataframes_cached(reload_data=reload_data, label=label, **kwargs)

    def get_data(self, filename):
        df = pd.read_csv(filename)
        return {"data_frame": df}
