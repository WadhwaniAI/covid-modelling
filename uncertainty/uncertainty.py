import os
import pdb
import argparse
import numpy as np
np.random.seed(10)
import pandas as pd
from tqdm import tqdm
from os.path import exists
import matplotlib.pyplot as plt

from mcmc import MCMC
from mcmc_utils import predict, get_state


def visualize(mcmc: MCMC, compartments: list, end_date: str = None): 
    data = pd.concat([mcmc.df_train, mcmc.df_val])
    result = predict(data, mcmc.chains, end_date)

    data.set_index("date", inplace=True)

    plt.figure(figsize=(15, 10))
    train_start_date = mcmc.df_train.iloc[-mcmc.fit_days, :]['date']
    train_end_date = mcmc.df_train.iloc[-1, :]['date']
    
    actual = data[data.index >= train_start_date]
    pred = result[result.index >= train_start_date]
    
    color = plt.cm.rainbow(np.linspace(0,1,len(compartments)))

    for i, bucket in enumerate(compartments):
        plt.plot(actual.index.array, actual[bucket].tolist(), c=color[i], marker='o', label='Actual {}'.format(bucket))
        plt.plot(pred.index.array, pred[bucket].tolist(), c=color[i], label='Estimated {}'.format(bucket))
        plt.plot(pred.index.array, pred['{}_low'.format(bucket)].tolist(), c=color[i], linestyle='dashdot')
        plt.plot(pred.index.array, pred['{}_high'.format(bucket)].tolist(), c=color[i], linestyle='dashdot')
    plt.axvline(x=train_end_date, c='k', linestyle='dashed', label='train ends')
    plt.xticks(rotation=90)
    plt.legend()
    plt.title("95% confidence intervals for {}, {}".format(mcmc.district, mcmc.state))
    plt.tight_layout()
    
    plt.savefig('./plots/{}_mcmc_confidence_intervals_{}_{}.png'.format(mcmc.timestamp, mcmc.district, mcmc.state))
    plt.show()

def main(district: str, end_date: str, chains:int, iters: int):
    os.makedirs('./plots', exist_ok=True)
    state = get_state(district)
    mcmc = MCMC(state = state, district = district, n_chains = chains, iters = iters, fit_days=10, test_days=5, fit2new=True)
    mcmc.run()

    compartments = ["total_infected"]
    visualize(mcmc, compartments, end_date=end_date)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Uncertainty Estimation with MCMC', allow_abbrev=False)
    parser.add_argument('-d', '--district', type=str, required=True, help = 'name of the district')
    parser.add_argument('-e', '--end_date', type=str, default=None, help = 'forecast end date')
    parser.add_argument('-c', '--chains', type=int, default=5, help = 'no. of mcmc chains')
    parser.add_argument('-i', '--iters', type=int, default=25000, help = 'no. of iterations for mcmc')
    (args, _) = parser.parse_known_args()
    main(args.district, args.end_date, args.chains, args.iters)
