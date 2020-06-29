import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from main.seir.optimiser import Optimiser


def get_state(district):
    state_map = {
        "Mumbai": "Maharashtra",
        "Pune": "Maharashtra",
        "Jaipur": "Rajasthan",
        "Bengaluru Urban": "Karnataka",
        "Bengaluru Rural": "Karnataka",
        "Ahmedabad": "Gujarat",
        "North Delhi": "Delhi",
        "South Delhi": "Delhi",
        "East Delhi": "Delhi",
        "West Delhi": "Delhi"
    }
    return state_map[district]
    
def get_PI(pred_dfs, date, key, multiplier=1.96):
    pred_samples = list()
    for df in pred_dfs:
        pred_samples.append(df.loc[date, key])
        
    mu = np.array(pred_samples).mean()
    sigma = np.array(pred_samples).std()
    low = mu - multiplier*sigma
    high = mu + multiplier*sigma
    return mu, low, high

def set_optimizer(data: pd.DataFrame):
    optimiser = Optimiser(use_mcmc=False)
    default_params = optimiser.init_default_params(data)
    return optimiser, default_params

def predict(data: pd.DataFrame, mcmc_runs: list, end_date: str = None) -> pd.DataFrame:

    optimiser, default_params = set_optimizer(data)
    
    combined_acc = list()
    for k, run in enumerate(mcmc_runs):
        burn_in = int(len(run) / 2)
        combined_acc += run[0][burn_in:]
        
    n_samples = 1000
    sample_indices = np.random.uniform(0, len(combined_acc), n_samples)
    
    pred_dfs = list()
    for i in tqdm(sample_indices):
        pred_dfs.append(optimiser.solve(combined_acc[int(i)], default_params, data, initialisation='intermediate', loss_indices=[0, -1], end_date=end_date))

    for df in pred_dfs:
        df.set_index('date', inplace=True)

    result = pred_dfs[0].copy()
    for col in result.columns:
        result["{}_low".format(col)] = ''
        result["{}_high".format(col)] = ''

    for date in tqdm(pred_dfs[0].index):
        for key in pred_dfs[0]:
            result.loc[date, key], result.loc[date, "{}_low".format(key)], result.loc[date, "{}_high".format(key)] = get_PI(pred_dfs, date, key)
    
    return result

def accumulate(dict_list):
    accumulator = defaultdict(int)
    for elt in dict_list:
        for key in elt:
            accumulator[key]+=elt[key]
    return accumulator

def divide(dictvar, num):
    return {key:dictvar[key]/num for key in dictvar}
            
def avg_sum_chain(chain):
    chain_sums_avg = accumulate(chain)
    return divide(chain_sums_avg, len(chain))

def avg_sum_multiple_chains(chain_sums_avg):
    multiple_chain_sums_avg = accumulate(chain_sums_avg)
    return divide(multiple_chain_sums_avg, len(chain_sums_avg))

def compute_B(multiple_chain_sums_avg, chain_sums_avg, n, m):
    B = defaultdict(int)
    for elt in chain_sums_avg:
        for key in elt:
            B[key] += np.square(elt[key] - multiple_chain_sums_avg[key])
    return divide(B, (m-1)/n)

def compute_W(split_chains, chain_sums_avg, n, m):
    s = []
    for j in range(m):
        s_j_sq = defaultdict(int)
        chain = split_chains[j]
        chain_sum_avg_j = chain_sums_avg[j]
        for i in range(n):
            chain_elt = chain[i]
            for key in chain_elt:
                s_j_sq[key] += np.square(chain_elt[key] - chain_sum_avg_j [key])
        s_j_sq = divide(s_j_sq, n - 1)
        s.append(s_j_sq)
    return (divide (accumulate(s),m))

def divide_dict(d1, d2):
    accumulator = defaultdict(int)
    for key in d1:
        accumulator[key] = d1[key]/d2[key]
    return accumulator