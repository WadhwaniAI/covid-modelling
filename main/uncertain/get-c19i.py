import sys
from argparse import ArgumentParser

import pandas as pd

arguments = ArgumentParser()
arguments.add_argument('--state')
arguments.add_argument('--district')
arguments.add_argument('--city')
arguments.add_argument('--outlook', type=int)
# arguments.add_argument('--workers', type=int)
args = arguments.parse_args()

dcol = 'Date Announced'
icol = 'total_infected'
locations = {
    'Detected City': 'city',
    'Detected District': 'district',
    'Detected State': 'state',
}
usecols = {
    dcol: 'date',
}

query = []
for (k, v) in locations.items():
    if hasattr(args, v):
        value = getattr(args, v)
        if value is not None:
            query.append('{}.str.lower() == "{}"'.format(v, value.lower()))
            usecols.update({k: v})
assert query

df = (pd
      .read_csv(sys.stdin,
                usecols=usecols,
                parse_dates=[dcol],
                dayfirst=True)
      .rename(columns=usecols)
      .query(' and '.join(query))
      .assign(**{icol: 1})
      .filter(items=['date', icol])
      .groupby('date')
      .sum()
      .cumsum()
      .resample('D')
      .ffill())
assert not df.empty

df.to_csv(sys.stdout)
