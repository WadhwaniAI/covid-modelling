import sys
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import constants

def relative(x):
    y = x['variable'].astype('datetime64[D]')
    diff = y - y.min()

    return diff.apply(lambda x: x.total_seconds() / constants.day)

arguments = ArgumentParser()
arguments.add_argument('--output', type=Path, required=True)
args = arguments.parse_args()

df = (pd.
      read_csv(sys.stdin)
      .melt()
      .assign(days=relative))
sns.lineplot(x='days',
             y='value',
             data=df)
plt.grid(True)
plt.savefig(args.output)
