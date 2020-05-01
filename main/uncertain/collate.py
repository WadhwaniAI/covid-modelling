import sys
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool

import pandas as pd

def func(args):
    return pd.read_csv(args)

arguments = ArgumentParser()
arguments.add_argument('--estimates', type=Path)
arguments.add_argument('--workers', type=int)
args = arguments.parse_args()

with Pool(args.workers) as pool:
    records = pool.imap_unordered(func, args.estimates.iterdir())
    pd.concat(records).to_csv(sys.stdout, index=False)
