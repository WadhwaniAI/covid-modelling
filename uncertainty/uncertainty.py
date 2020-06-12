import os
import pdb
import json
import argparse
import numpy as np
np.random.seed(10)
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from os.path import exists, join, splitext

from mcmc import MCMC
from mcmc_utils import predict, get_state


def visualize_forecasts(mcmc: MCMC, compartments: list, end_date: str = None, out_dir: str = ''): 
    data = pd.concat([mcmc.df_train, mcmc.df_val])
    result = predict(data, mcmc.chains, end_date)

    data.set_index("date", inplace=True)

    plt.figure(figsize=(15, 10))
    train_start_date = mcmc.df_train.iloc[mcmc.fit_start, :]['date']
    train_end_date = mcmc.df_train.iloc[-1, :]['date']
    
    actual = data[data.index >= train_start_date]
    pred = result[result.index >= train_start_date]
    
    color = {
            "total_infected": 'c',
            "recovered": 'g',
            "hospitalised": 'b',
            "deceased": 'r'
    }

    for i, bucket in enumerate(compartments):
        plt.plot(actual.index.array, actual[bucket].tolist(), c=color[bucket], marker='o', label='Actual {}'.format(bucket))
        plt.plot(pred.index.array, pred[bucket].tolist(), c=color[bucket], label='Estimated {}'.format(bucket))
        plt.plot(pred.index.array, pred['{}_low'.format(bucket)].tolist(), c=color[bucket], linestyle='dashdot')
        plt.plot(pred.index.array, pred['{}_high'.format(bucket)].tolist(), c=color[bucket], linestyle='dashdot')
    plt.axvline(x=train_end_date, c='k', linestyle='dashed', label='train ends')
    plt.xticks(rotation=90)
    plt.legend()
    plt.title("95% confidence intervals for {}, {}".format(mcmc.district, mcmc.state))
    plt.tight_layout()
    
    plt.savefig(join(out_dir, 'forecasts_{}_{}.png'.format(mcmc.district, mcmc.state)))

    plt.figure()

    for i, bucket in enumerate(compartments):
        plt.plot(actual.index.array, actual[bucket].tolist(), c=color[bucket], marker='o', label='Actual {}'.format(bucket))
        plt.plot(pred.index.array, pred[bucket].tolist(), c=color[bucket], label='Estimated {}'.format(bucket))
    plt.axvline(x=train_end_date, c='k', linestyle='dashed', label='train ends')
    plt.xticks(rotation=90)
    plt.legend()
    plt.title("mean fit for {}, {}".format(mcmc.district, mcmc.state))
    plt.tight_layout()
    
    plt.savefig(join(out_dir, 'mean_fit_{}_{}.png'.format(mcmc.district, mcmc.state)))

def plot_chains(mcmc: MCMC, out_dir: str):
    color = plt.cm.rainbow(np.linspace(0, 1, mcmc.n_chains))
    params = [*mcmc.prior_ranges.keys()]

    for param in params:
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)

        for i, chain in enumerate(mcmc.chains):
            df = pd.DataFrame(chain[0])
            samples = np.array(df[param])
            plt.scatter(list(range(len(samples))), samples, s=10, c=color[i].reshape(1,-1), label='chain {}'.format(i+1))

        plt.xlabel("iterations")
        plt.title("Accepted {} samples".format(param))
        plt.legend()

        plt.subplot(1, 2, 2)

        for i, chain in enumerate(mcmc.chains):
            df = pd.DataFrame(chain[1])
            samples = np.array(df[param])
            plt.scatter(list(range(len(samples))), samples, s=10, c=color[i].reshape(1,-1), label='chain {}'.format(i+1))

        plt.xlabel("iterations")
        plt.title("Rejected {} samples".format(param))
        plt.legend()

        plt.savefig(join(out_dir, '{}_{}_{}.png'.format(param, mcmc.district, mcmc.state)))

def main(config: str):
    cfg = json.load(open(join('cfg', config)))
    exp_name = splitext(config)[0]
    mcmc = MCMC(cfg)
    mcmc.run()
    sig = mcmc.timestamp.strftime("%d-%b-%Y (%H:%M:%S)")
    out_dir = join('plots', '{}_{}'.format(sig, exp_name))
    os.makedirs(out_dir, exist_ok=True)
    with open(join(out_dir, config), 'w') as outfile:
        json.dump(cfg, outfile)
    with open(join(out_dir, "R_hat.json"), 'w') as outfile:
        json.dump(mcmc.R_hat, outfile)

    plot_chains(mcmc, out_dir)
    visualize_forecasts(mcmc, cfg["compartments"], end_date=cfg["end_date"], out_dir=out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Uncertainty Estimation with MCMC', allow_abbrev=False)
    parser.add_argument('-c', '--config', type=str, required=True, help = 'name config file')
    (args, _) = parser.parse_known_args()
    main(args.config)

