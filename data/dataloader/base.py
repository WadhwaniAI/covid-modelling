from abc import ABC, abstractmethod

import os
import datetime
import pickle

class BaseLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def pull_dataframes(self):
        pass

    @abstractmethod
    def pull_dataframes_cached(self, reload_data=False, label=None, **kwargs):
        os.makedirs("../../misc/cache/", exist_ok=True)

        loader_key = self.__class__.__name__
        label = '' if label is None else f'_{label}'
        picklefn = "../../misc/cache/dataframes_ts_{today}_{loader_key}{label}.pkl".format(
            today=datetime.datetime.today().strftime("%d%m%Y"), loader_key=loader_key, label=label)
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
