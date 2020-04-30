import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool, Queue

import pandas as pd

from util import ModelParameter, Logger, dsplit
from optimiser import Optimiser

class Producer:
    def __init__(self, data):
        self.opt = Optimiser(data)

    def __call__(self, df):
        for i in df.itertuples(index=False):
            row = { x: getattr(i, x) for x in ModelParameter._fields }
            df = self.opt.solve(row)
            yield df['total_infected'].to_dict()

def func(incoming, outgoing, data):
    index_col = 'date'
    df = pd.read_csv(data,
                     index_col=index_col,
                     parse_dates=[index_col])
    producer = Producer(df)

    while True:
        (name, group) = incoming.get()
        Logger.info(name)
        outgoing.put(list(producer(group)))

arguments = ArgumentParser()
arguments.add_argument('--data', type=Path)
arguments.add_argument('--parameters', type=Path)
arguments.add_argument('--burn-in', type=int)
arguments.add_argument('--workers', type=int)
args = arguments.parse_args()

incoming = Queue()
outgoing = Queue()
initargs = (
    outgoing,
    incoming,
    args.data,
)

with Pool(args.workers, func, initargs):
    query = [
        'accept == 1',
        'step > {}'.format(args.burn_in),
    ]
    groups = (pd
              .read_csv(args.parameters)
              .query(' and '.join(query))
              .groupby('order', sort=False))
    for i in groups:
        outgoing.put(i)

    writer = None
    for i in range(groups.ngroups):
        results = incoming.get()
        if writer is None:
            head = results[0]
            writer = csv.DictWriter(sys.stdout, fieldnames=head.keys())
            writer.writeheader()
        writer.writerows(results)
