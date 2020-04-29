import sys
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

arguments = ArgumentParser()
arguments.add_argument('--output', type=Path, required=True)
args = arguments.parse_args()

df = pd.read_csv(sys.stdin).melt()
sns.lineplot(x='variable',
             y='value',
             data=df)
plt.savefig(args.output)
