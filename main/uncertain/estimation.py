import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool

import numpy as np
import pandas as pd

from util import Parameter, Logger, dsplit
from optimiser import Optimiser

class Theta(Parameter):
    @staticmethod
    def get(x, y):
        return np.random.uniform(x, y)

class Sigma(Parameter):
    @staticmethod
    def get(x, y):
        return y - x

#
#
#
class OptimizationContainer:
    def __init__(self, train, days=None):
        self.train = train
        self.fdays = days

        self.optimiser = Optimiser(self.train)
        if self.fdays is not None:
            self.fdays = pd.Timedelta(days=days)

    def __call__(self, theta):
        if not theta:
            raise ValueError()
        params = theta._asmodel()._asdict()
        return self.optimiser.solve(params)

    def fwin(self, df):
        if self.fdays is not None:
            marker = df.index.max() - self.fdays
            df = df.loc[str(marker):]

        return df['total_infected']

#
#
#
class Metropolis:
    def __init__(self, theta, sigma):
        self.theta = theta
        self.sigma = sigma

    def __iter__(self):
        return self

    def __next__(self):
        theta = self.propose()
        accepted = self.accept(theta)
        if accepted:
            self.theta = theta

        return { 'accept': accepted, **self.theta._asdict() }

    def propose(self):
        return self.theta.sample(self.sigma)

    def accept(self, proposed):
        raise NotImplementedError()

#
#
#
class LogAcceptor(Metropolis):
    def __init__(self, theta, sigma, optimizer):
        super().__init__(theta, sigma)

        self.optimizer = optimizer
        self.pi = np.sqrt(2 * np.pi)

    def __call__(self, theta):
        return self.likelihood(theta) + self.prior(theta)

    def likelihood(self, theta):
        try:
            pred = self.optimizer(theta)
        except ValueError:
            return -np.inf

        (pred, train) = map(self.optimizer.fwin, (pred, self.optimizer.train))
        a = len(train) * np.log(self.pi * theta.sigma)
        b = np.sum(((train - pred) ** 2) / (2 * theta.sigma ** 2))

        return -(a + b)

    def prior(self, theta):
        # return np.log(bool(theta))
        return 0 if theta else -np.inf

    def accept(self, theta):
        (old, new) = map(self, (self.theta, theta))
        decision = new > old
        if not decision:
            x = np.random.uniform(0, 1)
            decision = x < np.exp(new - old)

        return decision

def each(metropolis, steps, order):
    every = int(10 ** (np.floor(np.log10(steps)) - 1))

    for (i, j) in zip(range(steps), metropolis):
        if not i or i % every == 0:
            Logger.info('{} {}'.format(order, i))

        yield {
            'order': order,
            'step': i,
            **j,
        }

#
#
#
def func(args):
    (i, opts) = args

    index_col = 'date'
    df = pd.read_csv(opts.data,
                     index_col=index_col,
                     parse_dates=[index_col])
    (theta, sigma) = [ x.from_config(opts.config) for x in (Theta, Sigma) ]

    split = dsplit(df, opts.outlook - 1)
    optimizer = OptimizationContainer(split.train, opts.fit_days)
    metropolis = LogAcceptor(theta, sigma, optimizer)

    return list(each(metropolis, opts.steps, i))

arguments = ArgumentParser()
arguments.add_argument('--outlook', type=int)
arguments.add_argument('--fit-days', type=int)
arguments.add_argument('--steps', type=int)
arguments.add_argument('--starts', type=int)
arguments.add_argument('--data', type=Path)
arguments.add_argument('--config', type=Path)
arguments.add_argument('--workers', type=int)
args = arguments.parse_args()

with Pool(args.workers) as pool:
    writer = None
    starts = args.workers if args.starts is None else args.starts
    assert starts

    iterable = map(lambda x: (x, args), range(starts))
    for i in pool.imap_unordered(func, iterable):
        if writer is None:
            head = i[0]
            writer = csv.DictWriter(sys.stdout, fieldnames=head.keys())
            writer.writeheader()
        writer.writerows(i)
