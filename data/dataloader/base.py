from abc import ABC, abstractmethod

import os
import datetime
import pickle

class BaseLoader(ABC):
    """Base Dataloader for all other dataloaders

    Args:
        ABC (class): abstract class
    """
    def __init__(self):
        pass

    @abstractmethod
    def pull_dataframes(self, **kwargs):
        """Abstract Method for pulling dataframes from a given source
        """
        pass

    @abstractmethod
    def pull_dataframes_cached(self, reload_data=False, label=None, 
                               cache_dir="../../misc/cache/", **kwargs):
        """Method for running `pull_dataframes` if it hasn't been run even once today 
        and caching the output (as .pkl), and reading from the cached output otherwise

        Args:
            reload_data (bool, optional): If true, the caching aspect is ignored. Defaults to False.
            label (str, optional): Used for athena as there can be multiple caches for 
            different locations. Defaults to None.
            cache_dir (str, optional): Where the cache the output of `pull_dataframes`. 
            Defaults to "../../misc/cache/".

        Returns:
            dict: Dict of processed dataframes
        """
        os.makedirs(cache_dir, exist_ok=True)
        loader_key = self.__class__.__name__
        label = '' if label is None else f'_{label}'
        picklefn = "{cache_dir}dataframes_ts_{today}_{loader_key}{label}.pkl".format(
            cache_dir=cache_dir, today=datetime.datetime.today().strftime("%d%m%Y"), 
            loader_key=loader_key, label=label)
        if reload_data:
            print("pulling from source")
            dataframes = self.pull_dataframes(**kwargs)
        else:
            try:
                with open(picklefn, 'rb') as pickle_file:
                    dataframes = pickle.load(pickle_file)
                print(f'loading from {picklefn}')
            except:
                print("pulling from source")
                dataframes = self.pull_dataframes(**kwargs)
                with open(picklefn, 'wb+') as pickle_file:
                    pickle.dump(dataframes, pickle_file)
        return dataframes

    @abstractmethod
    def get_data(self, *args, **kwargs):
        """Abstract method serving as handshake between data and fitting modules
        """
        pass
