import sys
import csv
import itertools as it
import collections as cl
from string import Template
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
import requests
from shapely.geometry import Point

Location = cl.namedtuple('Location', 'state, country')

def measurements(row, ignore):
    for (k, v) in row.items():
        if k not in ignore:
            yield (pd.to_datetime(k), v)

def func(args):
    (row, dtype, index) = args

    keys = {
        'point': ('Long', 'Lat'),
        'location': ('Province/State', 'Country/Region'),
    }
    ignore = set(it.chain.from_iterable(keys.values()))

    pt = Point(*[ float(row[x]) for x in keys['point'] ])
    loc = Location(*[ row[x] for x in keys['location'] ])

    records = []
    for (k, v) in row.items():
        if k not in ignore:
            ln = dict(zip((index, dtype), (pd.to_datetime(k), v)))
            ln.update(loc._asdict())
            ln['geometry'] = pt.to_wkt()
            records.append(ln)

    return records

def sources(index):
    dtypes = [
        'confirmed',
        'deaths',
        'recovered',
    ]

    url = 'https://raw.githubusercontent.com'
    path = Path('CSSEGISandData',
                'COVID-19',
                'master',
                'csse_covid_19_data',
                'csse_covid_19_time_series')
    fname = Template('time_series_covid19_${dtype}_global')

    for i in dtypes:
        p = path.joinpath(fname.substitute(dtype=i)).with_suffix('.csv')
        r = requests.get('{}/{}'.format(url, p), stream=True)
        reader = csv.DictReader(r.iter_lines(decode_unicode=True))
        for row in reader:
            yield (row, i, index)

with Pool() as pool:
    index = 'date'
    iterable = sources(index)
    records = it.chain.from_iterable(pool.imap_unordered(func, iterable))
    pd.DataFrame.from_records(records, index=index).to_csv(sys.stdout)
