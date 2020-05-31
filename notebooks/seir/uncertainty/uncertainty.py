import os
import pdb
import numpy as np
np.random.seed(10)
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from os.path import exists, join

from mcmc import MCMC
from mcmc_utils import predict, get_PI


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

def main():
    os.makedirs('./plots', exist_ok=True)
    mcmc = MCMC(state = "Maharashtra", district = "Pune", n_chains = 5, iters = 100, fit_days=10, test_days=5)
    mcmc.run()

    compartments = ["total_infected"]
    visualize(mcmc, compartments, end_date="2020-06-24")


if __name__=="__main__":
    main()