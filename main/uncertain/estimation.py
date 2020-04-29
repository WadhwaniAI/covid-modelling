import sys
import csv
import collections as cl
from pathlib import Path
from argparse import ArgumentParser
from configparser import ConfigParser
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
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.optimiser = Optimiser(self.train)

    def __call__(self, theta):
        if not theta:
            raise ValueError()
        return self.optimiser.solve(theta._asdict(), self.train)

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
    infected = 'total_infected'

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
        pred = pred[self.infected]
        train = self.optimizer.train[self.infected]

        a = len(train) * np.log(self.pi * theta.sigma)
        b = np.sum(((train - pred) ** 2) / (2 * theta.sigma ** 2))

        return -(a + b)

    def prior(self, theta):
        # return np.log(bool(theta))
        return int(bool(theta))

    def accept(self, proposed):
        (old, new) = map(self, (self.theta, proposed))
        decision = new > old
        if not decision:
            x = np.random.uniform(0, 1)
            decision = x < np.exp(new - old)

        return decision

def each(metropolis, steps, order):
    for (i, j) in zip(range(steps), metropolis):
        yield {
            'order': order,
            'step': i,
            **j,
        }

#
#
#
def func(args):
    (i, data, config, outlook, steps) = args
    Logger.info(i)

    index_col = 'date'
    df = pd.read_csv(data, index_col=index_col, parse_dates=[index_col])
    (theta, sigma) = [ x.from_config(config) for x in (Theta, Sigma) ]
    optimizer = OptimizationContainer(*ttsplit(df, outlook - 1))
    metropolis = LogAcceptor(theta, sigma, optimizer)

    return list(each(metropolis, steps, i))

arguments = ArgumentParser()
arguments.add_argument('--outlook', type=int)
arguments.add_argument('--steps', type=int)
arguments.add_argument('--starts', type=int)
arguments.add_argument('--data', type=Path)
arguments.add_argument('--config', type=Path)
arguments.add_argument('--workers', type=int)
args = arguments.parse_args()

with Pool(args.workers) as pool:
    writer = None
    opts = (
        args.data,
        args.config,
        args.outlook,
        args.steps,
    )
    starts = args.workers if args.starts is None else args.starts
    assert starts

    iterable = map(lambda x: (x, *opts), range(starts))
    for i in pool.imap_unordered(func, iterable):
        if writer is None:
            head = i[0]
            writer = csv.DictWriter(sys.stdout, fieldnames=head.keys())
            writer.writeheader()
        writer.writerows(i)
