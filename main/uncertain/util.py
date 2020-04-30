import os
import logging
import collections as cl
from configparser import ConfigParser

import numpy as np
import pandas as pd

lvl = os.environ.get('PYTHONLOGLEVEL', 'WARNING').upper()
fmt = '[ %(asctime)s %(levelname)s %(process)d ] %(message)s'
logging.basicConfig(format=fmt,
                    datefmt="%d %H:%M:%S",
                    level=lvl)
logging.captureWarnings(True)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
Logger = logging.getLogger(__name__)

Split = cl.namedtuple('Split', 'train, test')

class CaseConfigParser(ConfigParser):
    def optionxform(self, optionstr):
        return optionstr

ModelParameter = cl.namedtuple('ModelParameter', [
    'R0',
    'T_inc',
    'T_inf',
    'T_recov_severe',
    'P_severe',
    'intervention_amount',
])

class Parameter(cl.namedtuple('Parameter', [
        *ModelParameter._fields,
        'sigma',
])):
    __slots__ = ()

    @classmethod
    def from_config(cls, config, key='model'):
        parser = CaseConfigParser()
        parser.read(config)

        p = {}
        for (k, v) in parser[key].items():
            p[k] = cls.get(*map(float, v.split(',')))

        return cls(**p)

    @staticmethod
    def get(x, y):
        raise NotImplementedError()

    def __add__(self, other):
        return type(self)(*np.random.normal(loc=self, scale=other))

    def __bool__(self):
        return all(x >= 0 for x in self)

    def _asmodel(self):
        params = map(lambda x: getattr(self, x), ModelParameter._fields)
        return ModelParameter(*params)

def dsplit(df, outlook):
    y = df.index.max() - pd.DateOffset(days=outlook)
    x = y - pd.DateOffset(days=1)

    return Split(df.loc[:str(x)], df.loc[str(y):])
