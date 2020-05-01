import sys
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool

import pandas as pd

def func(args):
    (path, burn, reject) = args

    df = pd.read_csv(path)
    df = df.iloc[burn:]
    if not reject:
        df = df.query('accept == 1')

    return df

arguments = ArgumentParser()
arguments.add_argument('--estimates', type=Path)
arguments.add_argument('--burn-in', type=int, default=0)
arguments.add_argument('--with-rejects', action='store_true')
arguments.add_argument('--workers', type=int)
args = arguments.parse_args()

with Pool(args.workers) as pool:
    estimates = args.estimates.iterdir()
    iterable = map(lambda x: (x, args.burn_in, args.with_rejects), estimates)
    records = pool.imap_unordered(func, iterable)
    pd.concat(records).to_csv(sys.stdout, index=False)
